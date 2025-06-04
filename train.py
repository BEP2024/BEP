import os
import time

import torch
import torch.nn as nn
import utils
from torch.autograd import Variable
from tqdm import tqdm
import time
import torch.nn.functional as F

import utils1.config as config
from gqa_data import GQAEvaluator

Tensor = torch.cuda.FloatTensor


def compute_supcon_loss(feats, qtype):
    tau = 1.0
    if isinstance(qtype, tuple):
        i = 0
        dic = {}
        for item in qtype:
            if item not in dic:
                dic[item] = i
                i = i + 1
        tau = 1.0
        qtype = torch.tensor([dic[item] for item in qtype]).cuda()
    feats_filt = F.normalize(feats, dim=1)
    targets_r = qtype.reshape(-1, 1)
    targets_c = qtype.reshape(1, -1)
    mask = targets_r == targets_c
    mask = mask.float().cuda()
    feats_sim = torch.exp(torch.matmul(feats_filt, feats_filt.T) / tau)
    negatives = feats_sim * (1.0 - mask)
    negative_sum = torch.sum(negatives)
    positives = torch.log(feats_sim / negative_sum) * mask
    positive_sum = torch.sum(positives)
    positive_sum = positive_sum / torch.sum(mask)

    sup_con_loss = -1 * torch.mean(positive_sum)
    return sup_con_loss


def compute_score_with_logits(logits, labels):
    logits = torch.argmax(logits, 1)
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores


def calc_genb_loss(logits, bias, labels):
    gen_grad = F.hardtanh(2 * labels * torch.sigmoid(-2 * labels * bias.detach()), 0, 1)
    loss = F.binary_cross_entropy_with_logits(logits, gen_grad)
    loss *= labels.size(1)
    return loss


def train_ce(model, m_model, loss_fn, genb, discriminator, train_loader, eval_loader, ce_loader, easy_loader, args,
             qid2type):
    torch.autograd.set_detect_anomaly(True)
    num_epochs = args.epochs
    run_eval = args.eval_each_epoch
    output = args.output
    optim = torch.optim.Adamax(
        [{'params': filter(lambda p: p.requires_grad, model.parameters())}, {'params': m_model.parameters()}], lr=0.001)

    optim_G = torch.optim.Adamax(filter(lambda p: p.requires_grad, genb.parameters()), lr=0.001)
    optim_D = torch.optim.Adamax(filter(lambda p: p.requires_grad, discriminator.parameters()), lr=0.001)
    logger = utils.Logger(os.path.join(output, 'log.txt'))
    total_step = 0
    best_eval_score = 0

    genb.train(True)
    discriminator.train(True)

    kld = nn.KLDivLoss(reduction='batchmean')
    bce = nn.BCELoss()

    for epoch in range(num_epochs):

        total_loss = 0
        train_score = 0

        t = time.time()
        for i, (v, q, a, qid, bias, mg, f1, type) in tqdm(enumerate(train_loader), ncols=100,
                                                          desc="Epoch %d" % (epoch + 1), total=len(train_loader)):
            total_step += 1

            #########################################
            v = Variable(v).cuda().requires_grad_()
            q = Variable(q).cuda()
            a = Variable(a).cuda()
            valid = Variable(Tensor(v.size(0), 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(v.size(0), 1).fill_(0.0), requires_grad=False)

            mg = mg.cuda()
            f1 = f1.cuda()
            bias = bias.cuda()
            gt = torch.argmax(a, 1)

            #########################################

            # get model output
            optim.zero_grad()
            hidden_, pred = model(v, q)

            hidden, pred1 = m_model(hidden_, pred, mg, epoch, a)

            dict_args = {'margin': mg, 'bias': bias, 'hidden': hidden, 'epoch': epoch, 'per': f1}

            # train genb
            optim_G.zero_grad()
            optim_D.zero_grad()

            pred_g = genb(v, q, gen=True)

            g_loss = F.binary_cross_entropy_with_logits(pred_g, a, reduction='none').mean()
            g_loss *= a.size(1)

            vae_preds = discriminator(pred_g)
            main_preds = discriminator(pred)

            g_distill = kld(pred_g, pred.detach())
            dsc_loss = bce(vae_preds, valid) + bce(main_preds, valid)

            g_loss = g_loss + dsc_loss + g_distill * 5

            g_loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(genb.parameters(), 0.25)

            # done training genb

            # train the discriminator
            dsc_loss = bce(vae_preds, fake) + bce(main_preds, valid)

            dsc_loss.backward(retain_graph=True)
            optim_G.step()
            optim_D.step()
            # done training the discriminator

            # use genb to train the robust model
            genb.train(False)
            pred_g = genb(v, q, gen=False)

            ce_loss = -F.log_softmax(pred, dim=-1) * a
            ce_loss = ce_loss * f1
            loss = ce_loss.sum(dim=-1).mean() + loss_fn(hidden, a, **dict_args)

            genb_loss = calc_genb_loss(pred, pred_g, a)

            gt = torch.argmax(a, 1)

            genb_loss = genb_loss + compute_supcon_loss(hidden_, gt) + loss

            genb_loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optim.step()

            genb.train(True)
            total_loss += genb_loss.item() * q.size(0)

            ce_logits = F.normalize(pred)
            pred_l = F.normalize(pred1)
            pred = (ce_logits + pred_l) / 2

            batch_score = compute_score_with_logits(pred, a.data).sum()
            train_score += batch_score

        total_loss /= len(train_loader.dataset)
        train_score = 100 * train_score / len(train_loader.dataset)

        logger.write('Epoch %d, time: %.2f' % (epoch + 1, time.time() - t))
        logger.write('\ttrain_loss: %.2f, score: %.2f' % (total_loss, train_score))

        if run_eval:
            model.train(False)
            results = evaluate_ce(model, m_model, eval_loader, ce_loader, easy_loader, qid2type)
            results["epoch"] = epoch
            results["step"] = total_step
            results["train_loss"] = total_loss
            results["train_score"] = train_score

            model.train(True)

            eval_score = results["score"]
            ce_score = results["ce_score"]
            easy_score = results["easy_score"]
            logger.write('\teval score: %.2f ce score: %.2f easy score: %.2f' % (
            100 * eval_score, 100 * ce_score, 100 * easy_score))

            main_eval_score = eval_score

            if main_eval_score > best_eval_score:
                model_path = os.path.join(output, 'model.pth')
                m_model_path = os.path.join(output, 'm_model.pth')
                genb_path = os.path.join(output, 'genb.pth')
                torch.save(model.state_dict(), model_path)
                torch.save(m_model.state_dict(), m_model_path)
                torch.save(genb.state_dict(), genb_path)
                best_eval_score = main_eval_score

        model_path = os.path.join(output, 'model_final.pth')
        torch.save(model.state_dict(), model_path)

    print('best eval score: %.2f' % (best_eval_score * 100))


def train_gqa(model, m_model, loss_fn, genb, discriminator, train_loader, eval_loader, args, qid2type):
    torch.autograd.set_detect_anomaly(True)
    num_epochs = args.epochs
    run_eval = args.eval_each_epoch
    output = args.output
    optim = torch.optim.Adamax(
        [{'params': filter(lambda p: p.requires_grad, model.parameters())}, {'params': m_model.parameters()}], lr=0.001)

    optim_G = torch.optim.Adamax(filter(lambda p: p.requires_grad, genb.parameters()), lr=0.001)
    optim_D = torch.optim.Adamax(filter(lambda p: p.requires_grad, discriminator.parameters()), lr=0.001)
    logger = utils.Logger(os.path.join(output, 'log.txt'))
    total_step = 0
    best_eval_score = 0
    skipped_batches = 0

    genb.train(True)
    discriminator.train(True)

    kld = nn.KLDivLoss(reduction='batchmean')
    bce = nn.BCELoss()

    for epoch in range(num_epochs):
        total_loss = 0
        train_score = 0
        epoch_skipped = 0

        t = time.time()
        for i, (v, q, a, qid, bias, mg, f1, type) in tqdm(enumerate(train_loader), ncols=100,
                                                          desc="Epoch %d" % (epoch + 1), total=len(train_loader)):
            total_step += 1

            #########################################
            v = Variable(v).cuda().requires_grad_()
            q = Variable(q).cuda()
            a = Variable(a).cuda()
            valid = Variable(Tensor(v.size(0), 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(v.size(0), 1).fill_(0.0), requires_grad=False)

            mg = mg.cuda()
            f1 = f1.cuda()
            bias = bias.cuda()
            gt = torch.argmax(a, 1)

            # 确保所有模型都在训练模式
            model.train()
            m_model.train()
            genb.train()
            discriminator.train()

            # 检查输入数据是否包含NaN
            if torch.isnan(v).any() or torch.isnan(q).any() or torch.isnan(a).any() or torch.isnan(
                    mg).any() or torch.isnan(f1).any() or torch.isnan(bias).any():
                print(f"Warning: NaN detected in input data at epoch {epoch + 1}, batch {i}, skipping...")
                epoch_skipped += 1
                skipped_batches += 1
                continue

            # get model output
            optim.zero_grad()
            hidden_, pred = model(v, q)

            # 检查模型输出是否包含NaN
            if torch.isnan(hidden_).any() or torch.isnan(pred).any():
                print(f"Warning: NaN detected in model output at epoch {epoch + 1}, batch {i}, skipping...")
                epoch_skipped += 1
                skipped_batches += 1
                continue

            hidden, pred1 = m_model(hidden_, pred, mg, epoch, a)

            # 检查m_model输出是否包含NaN
            if torch.isnan(hidden).any() or torch.isnan(pred1).any():
                print(f"Warning: NaN detected in m_model output at epoch {epoch + 1}, batch {i}, skipping...")
                epoch_skipped += 1
                skipped_batches += 1
                continue

            dict_args = {'margin': mg, 'bias': bias, 'hidden': hidden, 'epoch': epoch, 'per': f1}

            # train genb
            optim_G.zero_grad()
            optim_D.zero_grad()

            pred_g = genb(v, q, gen=True)

            # 检查genb输出是否包含NaN
            if torch.isnan(pred_g).any():
                print(f"Warning: NaN detected in genb output at epoch {epoch + 1}, batch {i}, skipping...")
                epoch_skipped += 1
                skipped_batches += 1
                continue

            g_loss = F.binary_cross_entropy_with_logits(pred_g, a, reduction='none').mean()
            g_loss *= a.size(1)

            vae_preds = discriminator(pred_g)
            main_preds = discriminator(pred)

            # 检查discriminator输出是否包含NaN
            if torch.isnan(vae_preds).any() or torch.isnan(main_preds).any():
                print(f"Warning: NaN detected in discriminator output at epoch {epoch + 1}, batch {i}, skipping...")
                epoch_skipped += 1
                skipped_batches += 1
                continue

            g_distill = kld(pred_g, pred.detach())
            dsc_loss = bce(vae_preds, valid) + bce(main_preds, valid)

            g_loss = g_loss + dsc_loss + g_distill * 5

            # 检查g_loss是否包含NaN
            if torch.isnan(g_loss) or torch.isinf(g_loss):
                print(f"Warning: NaN/Inf detected in g_loss at epoch {epoch + 1}, batch {i}, skipping...")
                epoch_skipped += 1
                skipped_batches += 1
                continue

            g_loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(genb.parameters(), 0.25)

            # train the discriminator
            dsc_loss = bce(vae_preds, fake) + bce(main_preds, valid)

            # 检查dsc_loss是否包含NaN
            if torch.isnan(dsc_loss) or torch.isinf(dsc_loss):
                print(f"Warning: NaN/Inf detected in dsc_loss at epoch {epoch + 1}, batch {i}, skipping...")
                epoch_skipped += 1
                skipped_batches += 1
                continue

            dsc_loss.backward(retain_graph=True)
            optim_G.step()
            optim_D.step()

            # use genb to train the robust model
            genb.train(False)
            pred_g = genb(v, q, gen=False)

            # 再次检查genb输出
            if torch.isnan(pred_g).any():
                print(
                    f"Warning: NaN detected in genb output (second pass) at epoch {epoch + 1}, batch {i}, skipping...")
                epoch_skipped += 1
                skipped_batches += 1
                continue

            ce_loss = -F.log_softmax(pred, dim=-1) * a
            ce_loss = ce_loss * f1
            loss = ce_loss.sum(dim=-1).mean() + loss_fn(hidden, a, **dict_args)

            # 检查loss是否包含NaN
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN/Inf detected in loss at epoch {epoch + 1}, batch {i}, skipping...")
                epoch_skipped += 1
                skipped_batches += 1
                continue

            genb_loss = calc_genb_loss(pred, pred_g, a)
            gt = torch.argmax(a, 1)
            genb_loss = genb_loss + compute_supcon_loss(hidden_, gt) + loss

            # 检查最终的genb_loss是否包含NaN
            if torch.isnan(genb_loss) or torch.isinf(genb_loss):
                print(f"Warning: NaN/Inf detected in final genb_loss at epoch {epoch + 1}, batch {i}, skipping...")
                epoch_skipped += 1
                skipped_batches += 1
                continue

            # 在反向传播之前检查梯度
            try:
                genb_loss.backward()
            except RuntimeError as e:
                if "nan" in str(e).lower():
                    print(f"Warning: NaN detected during backward pass at epoch {epoch + 1}, batch {i}, skipping...")
                    epoch_skipped += 1
                    skipped_batches += 1
                    continue
                else:
                    raise e

            nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optim.step()

            genb.train(True)
            total_loss += genb_loss.item() * q.size(0)

            ce_logits = F.normalize(pred)
            pred_l = F.normalize(pred1)
            pred = (ce_logits + pred_l) / 2

            batch_score = compute_score_with_logits(pred, a.data).sum()
            train_score += batch_score

        total_loss /= len(train_loader.dataset)
        train_score = 100 * train_score / len(train_loader.dataset)

        logger.write('Epoch %d, time: %.2f' % (epoch + 1, time.time() - t))
        logger.write('\ttrain_loss: %.2f, score: %.2f' % (total_loss, train_score))
        logger.write('\tskipped batches in this epoch: %d, total skipped: %d' % (epoch_skipped, skipped_batches))

        if run_eval:
            model.train(False)
            evaluator = GQAEvaluator(eval_loader.dataset.raw_dataset)

            # 收集验证集的预测结果
            quesid2ans = {}
            with torch.no_grad():
                for v, q, a, qids, bias, mg, f1, type in tqdm(eval_loader):
                    v = Variable(v, requires_grad=False).cuda()
                    q = Variable(q, requires_grad=False).cuda()
                    mg = Variable(mg, requires_grad=False).cuda()
                    hidden_, pred = model(v, q)
                    hidden, pred_m = m_model(hidden_, pred, mg, 0, a)

                    pred = F.softmax(F.normalize(pred) / config.temp, 1)
                    pred_m = F.softmax(F.normalize(pred_m), 1)
                    pred = config.alpha * pred_m + (1 - config.alpha) * pred

                    pred_ans = torch.argmax(pred, dim=1)
                    for i, qid in enumerate(qids):
                        ans = eval_loader.dataset.label2ans[pred_ans[i].item()]
                        if isinstance(qid, str):
                            quesid2ans[qid] = ans
                        else:
                            quesid2ans[qid.item()] = ans

            # 使用评估器计算分数
            main_eval_score = evaluator.evaluate(quesid2ans)

            # 保存预测结果到文件
            result_path = os.path.join(output, f'predictions_epoch_{epoch + 1}.json')
            evaluator.dump_result(quesid2ans, result_path)

            model.train(True)

            logger.write('\toverall score: %.2f' % (100 * main_eval_score))

            # 保存每个epoch的模型权重
            epoch_weights_dir = os.path.join(output, f'epoch_{epoch + 1}')
            os.makedirs(epoch_weights_dir, exist_ok=True)

            model_path = os.path.join(epoch_weights_dir, 'model.pth')
            m_model_path = os.path.join(epoch_weights_dir, 'm_model.pth')
            genb_path = os.path.join(epoch_weights_dir, 'genb.pth')
            torch.save(m_model.state_dict(), m_model_path)
            torch.save(model.state_dict(), model_path)
            torch.save(genb.state_dict(), genb_path)

            if main_eval_score > best_eval_score:
                model_path = os.path.join(output, 'model.pth')
                m_model_path = os.path.join(output, 'm_model.pth')
                genb_path = os.path.join(output, 'genb.pth')
                torch.save(m_model.state_dict(), m_model_path)
                torch.save(model.state_dict(), model_path)
                torch.save(genb.state_dict(), genb_path)
                best_eval_score = main_eval_score

        model_path = os.path.join(output, 'model_final.pth')
        torch.save(model.state_dict(), model_path)
    print('best eval score: %.2f' % (best_eval_score * 100))


def train(model, m_model, loss_fn, genb, discriminator, train_loader, eval_loader, args, qid2type):
    torch.autograd.set_detect_anomaly(True)
    num_epochs = args.epochs
    run_eval = args.eval_each_epoch
    output = args.output
    optim = torch.optim.Adamax(
        [{'params': filter(lambda p: p.requires_grad, model.parameters())}, {'params': m_model.parameters()}], lr=0.001)

    optim_G = torch.optim.Adamax(filter(lambda p: p.requires_grad, genb.parameters()), lr=0.001)
    optim_D = torch.optim.Adamax(filter(lambda p: p.requires_grad, discriminator.parameters()), lr=0.001)
    logger = utils.Logger(os.path.join(output, 'log.txt'))
    total_step = 0
    best_eval_score = 0

    genb.train(True)
    discriminator.train(True)

    kld = nn.KLDivLoss(reduction='batchmean')
    bce = nn.BCELoss()

    for epoch in range(num_epochs):
        total_loss = 0
        train_score = 0

        t = time.time()
        for i, (v, q, a, qid, bias, mg, f1, type) in tqdm(enumerate(train_loader), ncols=100,
                                                          desc="Epoch %d" % (epoch + 1), total=len(train_loader)):
            total_step += 1

            #########################################
            v = Variable(v).cuda().requires_grad_()
            q = Variable(q).cuda()
            a = Variable(a).cuda()
            valid = Variable(Tensor(v.size(0), 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(v.size(0), 1).fill_(0.0), requires_grad=False)

            mg = mg.cuda()
            f1 = f1.cuda()
            bias = bias.cuda()
            gt = torch.argmax(a, 1)

            #########################################

            # get model output
            optim.zero_grad()
            hidden_, pred = model(v, q)
            hidden, pred1 = m_model(hidden_, pred, mg, epoch, a)
            dict_args = {'margin': mg, 'bias': bias, 'hidden': hidden, 'epoch': epoch, 'per': f1}

            # train genb
            optim_G.zero_grad()
            optim_D.zero_grad()

            pred_g = genb(v, q, gen=True)
            g_loss = F.binary_cross_entropy_with_logits(pred_g, a, reduction='none').mean()
            g_loss *= a.size(1)

            vae_preds = discriminator(pred_g)
            main_preds = discriminator(pred)

            g_distill = kld(pred_g, pred.detach())
            dsc_loss = bce(vae_preds, valid) + bce(main_preds, valid)

            g_loss = g_loss + dsc_loss + g_distill * 5
            g_loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(genb.parameters(), 0.25)

            # done training genb

            # train the discriminator
            dsc_loss = bce(vae_preds, fake) + bce(main_preds, valid)
            dsc_loss.backward(retain_graph=True)
            optim_G.step()
            optim_D.step()
            # done training the discriminator

            # use genb to train the robust model
            genb.train(False)
            pred_g = genb(v, q, gen=False)

            ce_loss = -F.log_softmax(pred, dim=-1) * a
            ce_loss = ce_loss * f1
            loss = ce_loss.sum(dim=-1).mean() + loss_fn(hidden, a, **dict_args)

            genb_loss = calc_genb_loss(pred, pred_g, a)

            gt = torch.argmax(a, 1)

            genb_loss = genb_loss + compute_supcon_loss(hidden_, gt) + loss
            genb_loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optim.step()

            genb.train(True)
            total_loss += genb_loss.item() * q.size(0)

            ce_logits = F.normalize(pred)
            pred_l = F.normalize(pred1)
            pred = (ce_logits + pred_l) / 2

            batch_score = compute_score_with_logits(pred, a.data).sum()
            train_score += batch_score

        total_loss /= len(train_loader.dataset)
        train_score = 100 * train_score / len(train_loader.dataset)

        logger.write('Epoch %d, time: %.2f' % (epoch + 1, time.time() - t))
        logger.write('\ttrain_loss: %.2f, score: %.2f' % (total_loss, train_score))

        if run_eval:
            model.train(False)
            results = evaluate(model, m_model, eval_loader, qid2type)
            results["epoch"] = epoch
            results["step"] = total_step
            results["train_loss"] = total_loss
            results["train_score"] = train_score

            model.train(True)

            eval_score = results["score"]
            bound = results["upper_bound"]
            yn = results['score_yesno']
            other = results['score_other']
            num = results['score_number']
            logger.write('\teval score: %.2f (%.2f)' % (100 * eval_score, 100 * bound))
            logger.write('\tyn score: %.2f other score: %.2f num score: %.2f' % (100 * yn, 100 * other, 100 * num))

            main_eval_score = eval_score

            if main_eval_score > best_eval_score:
                model_path = os.path.join(output, 'model.pth')
                m_model_path = os.path.join(output, 'm_model.pth')
                genb_path = os.path.join(output, 'genb.pth')
                torch.save(m_model.state_dict(), m_model_path)
                torch.save(model.state_dict(), model_path)
                torch.save(genb.state_dict(), genb_path)
                best_eval_score = main_eval_score

        model_path = os.path.join(output, 'model_final.pth')
        torch.save(model.state_dict(), model_path)
    print('best eval score: %.2f' % (best_eval_score * 100))


def evaluate(model, m_model, dataloader, qid2type):
    score = 0
    upper_bound = 0
    score_yesno = 0
    score_number = 0
    score_other = 0
    total_yesno = 0
    total_number = 0
    total_other = 0

    for v, q, a, qids, bias, mg, f1, type in tqdm(dataloader, ncols=100, total=len(dataloader), desc="eval"):
        v = Variable(v, requires_grad=False).cuda()
        q = Variable(q, requires_grad=False).cuda()
        mg = mg.cuda()
        hidden_, pred = model(v, q)
        hidden, pred_m = m_model(hidden_, pred, mg, 0, a)

        pred = F.softmax(F.normalize(pred) / config.temp, 1)
        pred_m = F.softmax(F.normalize(pred_m), 1)
        pred = config.alpha * pred_m + (1 - config.alpha) * pred

        batch_score = compute_score_with_logits(pred, a.cuda()).cpu().numpy().sum(1)
        score += batch_score.sum()
        upper_bound += (a.max(1)[0]).sum()
        qids = qids.detach().cpu().int().numpy()
        for j in range(len(qids)):
            qid = qids[j]
            typ = qid2type[str(qid)]
            if typ == 'yes/no':
                score_yesno += batch_score[j]
                total_yesno += 1
            elif typ == 'other':
                score_other += batch_score[j]
                total_other += 1
            elif typ == 'number':
                score_number += batch_score[j]
                total_number += 1
            else:
                print('Hahahahahahahahahahaha')

    score = score / len(dataloader.dataset)
    upper_bound = upper_bound / len(dataloader.dataset)
    score_yesno /= total_yesno
    score_other /= total_other
    score_number /= total_number

    results = dict(
        score=score,
        upper_bound=upper_bound,
        score_yesno=score_yesno,
        score_other=score_other,
        score_number=score_number,
    )
    return results


def evaluate_ce(model, m_model, dataloader, ce_loader, easy_loader, qid2type):
    score = 0
    ce_score = 0
    easy_score = 0
    upper_bound = 0
    score_yesno = 0
    score_number = 0
    score_other = 0
    total_yesno = 0
    total_number = 0
    total_other = 0

    for v, q, a, qids, bias, mg, f1, type in tqdm(dataloader, ncols=100, total=len(dataloader), desc="eval"):
        v = Variable(v, requires_grad=False).cuda()
        q = Variable(q, requires_grad=False).cuda()
        mg = mg.cuda()
        hidden_, pred = model(v, q)
        hidden, pred_m = m_model(hidden_, pred, mg, 0, a)

        pred = F.softmax(F.normalize(pred) / config.temp, 1)
        pred_m = F.softmax(F.normalize(pred_m), 1)
        pred = config.alpha * pred_m + (1 - config.alpha) * pred

        batch_score = compute_score_with_logits(pred, a.cuda()).cpu().numpy().sum(1)
        score += batch_score.sum()
        upper_bound += (a.max(1)[0]).sum()
        qids = qids.detach().cpu().int().numpy()
        for j in range(len(qids)):
            qid = qids[j]
            typ = qid2type[str(qid)]
            if typ == 'yes/no':
                score_yesno += batch_score[j]
                total_yesno += 1
            elif typ == 'other':
                score_other += batch_score[j]
                total_other += 1
            elif typ == 'number':
                score_number += batch_score[j]
                total_number += 1
            else:
                print('Hahahahahahahahahahaha')
    for v, q, a, qids, bias, mg, f1, type in tqdm(ce_loader, ncols=100, total=len(dataloader), desc="eval"):
        v = Variable(v, requires_grad=False).cuda()
        q = Variable(q, requires_grad=False).cuda()
        mg = mg.cuda()
        hidden_, pred = model(v, q)
        hidden, pred_m = m_model(hidden_, pred, mg, 0, a)

        pred = F.softmax(F.normalize(pred) / config.temp, 1)
        pred_m = F.softmax(F.normalize(pred_m), 1)
        pred = config.alpha * pred_m + (1 - config.alpha) * pred

        batch_score = compute_score_with_logits(pred, a.cuda()).cpu().numpy().sum(1)
        ce_score += batch_score.sum()
    for v, q, a, qids, bias, mg, f1, type in tqdm(easy_loader, ncols=100, total=len(dataloader), desc="eval"):
        v = Variable(v, requires_grad=False).cuda()
        q = Variable(q, requires_grad=False).cuda()
        mg = mg.cuda()
        hidden_, pred = model(v, q)
        hidden, pred_m = m_model(hidden_, pred, mg, 0, a)

        pred = F.softmax(F.normalize(pred) / config.temp, 1)
        pred_m = F.softmax(F.normalize(pred_m), 1)
        pred = config.alpha * pred_m + (1 - config.alpha) * pred

        batch_score = compute_score_with_logits(pred, a.cuda()).cpu().numpy().sum(1)
        easy_score += batch_score.sum()

    score = score / len(dataloader.dataset)
    ce_score = ce_score / len(ce_loader.dataset)
    easy_score = easy_score / len(easy_loader.dataset)
    upper_bound = upper_bound / len(dataloader.dataset)
    score_yesno /= total_yesno
    score_other /= total_other
    score_number /= total_number

    results = dict(
        score=score,
        ce_score=ce_score,
        easy_score=easy_score,
        upper_bound=upper_bound,
        score_yesno=score_yesno,
        score_other=score_other,
        score_number=score_number,
    )
    return results
