import torch
import torch.nn.functional as F
from autoattack import AutoAttack
from autoattack.other_utils import Logger

import autopgd_train_clean
import utils_perceptual_eval


# Tools to attack 2AFC task.

def binary_margin_loss(logits, y):

    y_onehot = F.one_hot(y, num_classes=2)
    loss = logits * (1 - y_onehot) - logits * y_onehot
    return loss.sum(-1)


def eval(fp, loader, n_ex=10, bs=10, device='cuda:0', norm='Linf', eps=8/255, loss=None,
    n_iter=10, use_rs=False, attack_name='apgd', n_restarts=1, log_path=None,
    alpha_init=None, const_step_size=False, grad_type=None):
    """Run attack on 2AFC model."""

    logger = Logger(log_path)

    # Compute clean performance and embedding for all images.
    output = utils_perceptual_eval.get_emb(fp, loader, device=device, n_ex=n_ex,)
    stats, clean_acc, _ = utils_perceptual_eval.get_sim(output, logger)

    emb_ref = torch.cat([item[0] for item in output], dim=0)
    emb_left = torch.cat([item[1] for item in output], dim=0)
    emb_right = torch.cat([item[2] for item in output], dim=0)
    y = torch.stack([torch.tensor(item[3].astype(int), dtype=torch.int64) for item in output], dim=0)
    print(emb_ref.shape, emb_left.shape, emb_right.shape, y.shape)
    del emb_ref

    # Run attacks.
    x_test = []
    x_adv = []
    l_acc = []
    acc = torch.zeros([0])

    for i, (x_ref, x_left, x_right, lab, id) in enumerate(loader):
        x_test.append(x_ref)

        if (i + 1) % bs == 0 or i + 1 == n_ex:
            idx = torch.arange(acc.shape[0], i + 1).long()
            print(idx)

            clf_fn = lambda x: utils_perceptual_eval.clf(
                fp, x, emb_left[idx].to(device), emb_right[idx].to(device))

            if attack_name == 'apgd':
                out = autopgd_train_clean.apgd_train(
                    model=clf_fn,
                    x=torch.stack(x_test, dim=0).to(device),
                    y=y[idx].to(device),
                    norm=norm,
                    eps=eps,
                    n_iter=n_iter,
                    use_rs=use_rs,
                    loss=binary_margin_loss if loss is None else loss,
                    verbose=True,
                    is_train=False,
                    early_stop=True,
                    alpha_init=alpha_init,
                    const_step_size=const_step_size,
                    grad_type=grad_type,
                )
                x_adv.append(out[-1].cpu())
                l_acc.append(out[1].cpu().float())
                acc = torch.cat(l_acc, 0) if len(l_acc) > 1 else out[1].cpu().float()

            elif attack_name == 'apgd-largereps':
                out = autopgd_train_clean.apgd_largereps(
                    model=clf_fn,
                    x=torch.stack(x_test, dim=0).to(device),
                    y=y[idx].to(device),
                    norm=norm,
                    eps=eps,
                    n_iter=n_iter,
                    use_rs=use_rs,
                    loss=binary_margin_loss if loss is None else loss,
                    verbose=True,
                    is_train=False,
                    early_stop=True,
                    alpha_init=alpha_init,
                    const_step_size=const_step_size,
                )
                x_adv.append(out[-1].cpu())
                l_acc.append(out[1].cpu().float())
                acc = torch.cat(l_acc, 0) if len(l_acc) > 1 else out[1].cpu().float()

            elif attack_name in ['aa-apgd']:
                # FIX: this doesn't run because the alternative embeddings are fixed
                # while the attack uses only robust points.
                adversary = AutoAttack(clf_fn, norm=norm, eps=eps,
                    version='standard', seed=None, log_path=log_path,
                    device=device)
                adversary.attacks_to_run = ['apgd-ce']
                adversary.apgd.n_iter = n_iter
                adversary.apgd.n_restarts = n_restarts
                x_adv_curr = adversary.run_standard_evaluation(
                    torch.stack(x_test, dim=0).to(device), y[idx].to(device),
                    bs=bs)
                x_adv.append(x_adv_curr[-1].cpu())
                acc = torch.ones([1]) * -1

            else:
                raise ValueError(f'Unknown attack: {attack_name}.')

            x_test = []
            logger.log(f'batch={(i + 1) // bs:.0f} acc={acc.mean():.2%} ({acc.sum():.0f}/{acc.shape[0]})')

        if i + 1 == n_ex:
            break

    return torch.cat(x_adv, dim=0), acc


def eval_restarts(*args, attacks=['apgd-ce'], **kwargs):
    """Run multiple restarts of different attacks."""

    n_restarts = kwargs.get('n_restarts', 1)
    kwargs['n_restarts'] = 1
    log_path = kwargs.get('log_path', None)
    logger = Logger(log_path)
    n_ex = kwargs.get('n_ex', None)
    acc = torch.ones(n_ex)
    x_adv = None

    for r in range(n_restarts):

        for attack in attacks:

            if attack in ['apgd-ce', 'apgd-binary-margin']:
                loss = 'ce' if 'ce' in attack else 'binary-margin'
                kwargs['attack_name'] = 'apgd'
                kwargs['loss'] = loss
                x_adv_curr, acc_curr = eval(*args, **kwargs)

            else:
                raise ValueError(f'Unknown attack: {attack}.')

            if x_adv is None:
                x_adv = x_adv_curr.clone()
            else:
                # Update with successful attacks.
                succs = acc_curr == 0
                x_adv[succs] = x_adv_curr[succs].clone()
                acc = torch.min(acc, acc_curr)

            logger.log((f'restart={r} attack={attack}'
                f' acc={acc.mean():.2%}'
                ))

    return x_adv


# Same attacks as above but keeping the laoder structure.

def eval_loader(fp, loader, n_ex=10, bs=10, device='cuda:0', norm='Linf', eps=8/255, loss=None,
    n_iter=10, use_rs=False, attack_name='apgd', n_restarts=1, log_path=None,
    alpha_init=None, const_step_size=False, grad_type=None, metric_type='embedding'):
    """Run attack on 2AFC model."""

    logger = Logger(log_path)

    x_adv = []
    l_acc = []
    clean_acc = 0
    #acc = 0.
    seen_imgs = 0

    for i, batch in enumerate(loader):
        
        x_ref, x_left, x_right, lab, id =  batch
        #print(lab)
        bs = x_ref.shape[0]
        
        # Compute clean performance and embedding for a single batch.
        if metric_type == 'embedding':
            output = utils_perceptual_eval.get_emb(fp, [batch,], device=device, n_ex=-1)
            _, clean_acc_curr, _ = utils_perceptual_eval.get_sim(
                output, logger, verbose=False)
            emb_left = output[0][1].to(device)
            emb_right = output[0][2].to(device)
            clf_fn = lambda x: utils_perceptual_eval.clf(fp, x, emb_left, emb_right)
        elif metric_type == 'lpips':
            clean_acc_curr = utils_perceptual_eval.get_acc(
                fp, [batch,], device=device, n_ex=-1, mode='lpips', logger=logger)
            clf_fn = lambda x: utils_perceptual_eval.lpips_clf(
                fp, x, x_left.to(device), x_right.to(device), normalize=True)
            
        
        if attack_name == 'apgd':
            out = autopgd_train_clean.apgd_train(
                model=clf_fn,
                x=x_ref.to(device),
                y=lab.long().to(device), #torch.tensor(lab, dtype=torch.int64).to(device),
                norm=norm,
                eps=eps,
                n_iter=n_iter,
                use_rs=use_rs,
                loss=binary_margin_loss if loss is None else loss,
                verbose=True,
                is_train=False,
                early_stop=True,
                alpha_init=alpha_init,
                const_step_size=const_step_size,
                grad_type=grad_type,
            )
            x_adv.append(out[-1].cpu())
            l_acc.append(out[1].float().cpu())
            acc = torch.cat(l_acc, 0) if len(l_acc) > 1 else out[1].cpu().float()

        elif attack_name == 'square':
            import square

            attack_fn = square.SquareAttack(
                clf_fn,
                norm=norm,
                n_queries=n_iter,
                eps=eps,
                p_init=.8,
                n_restarts=1,
                seed=0,
                verbose=True,
                targeted=False,
                loss=loss,
                resc_schedule=True,
                device=device,
                early_stop=False,  # Otherwise left and right embedding should be changed.
            )
            out = attack_fn.perturb(x_ref.to(device), lab.long().to(device))
            x_adv.append(out[0].cpu())
            l_acc.append(out[1].float().cpu())
            acc = torch.cat(l_acc, 0) if len(l_acc) > 1 else out[1].cpu().float()

        else:
            raise ValueError(f'Unknown attack: {attack_name}.')

        seen_imgs += bs
        clean_acc += (clean_acc_curr[0] * bs)
        logger.log(f'batch={i + 1:.0f}'
                   f' [acc] clean={clean_acc / seen_imgs:.2%} ({clean_acc:.0f}/{seen_imgs})'
                   f' rob={acc.mean():.2%} ({acc.sum():.0f}/{acc.shape[0]})')

        if seen_imgs >= n_ex:
            break

    return x_adv, l_acc


def eval_restarts_loader(*args, attacks=['apgd-ce'], **kwargs):
    """Run multiple restarts of different attacks."""

    n_restarts = kwargs.get('n_restarts', 1)
    #kwargs['n_restarts'] = 1
    log_path = kwargs.get('log_path', None)
    logger = Logger(log_path)
    #n_ex = kwargs.get('n_ex', None)
    #acc = torch.ones(n_ex)
    #x_adv = []

    for r in range(n_restarts):

        for attack in attacks:

            if attack in ['apgd-ce', 'apgd-binary-margin']:
                loss = 'ce' if 'ce' in attack else 'binary-margin'
                kwargs['attack_name'] = 'apgd'
                kwargs['loss'] = loss
                x_adv_curr, acc_curr = eval_loader(*args, **kwargs)

            elif attack == 'square':
                loss = 'margin' #'ce' if 'ce' in attack else 'binary-margin'
                kwargs['attack_name'] = 'square'
                kwargs['loss'] = loss
                x_adv_curr, acc_curr = eval_loader(*args, **kwargs)

            else:
                raise ValueError(f'Unknown attack: {attack}.')

            if r == 0:
                # Just keep the output of first attack.
                x_adv = x_adv_curr.copy()
                acc = acc_curr.copy()
                acc_set = 0.
                seen_imgs = 0
                for item in acc:
                    acc_set += item.float().sum()
                    seen_imgs += item.shape[0]
            else:
                # Update each batch with successful attacks.
                acc_set = 0.
                seen_imgs = 0
                for e in range(len(x_adv)):
                    succs = acc_curr[e] == 0
                    x_adv[e][succs] = x_adv_curr[e][succs].clone()
                    acc[e] = torch.min(acc[e], acc_curr[e])
                    acc_set += acc[e].float().sum()
                    seen_imgs += x_adv[0].shape[0]

            logger.log((f'restart={r} attack={attack}'
                f' acc={acc_set / seen_imgs:.2%} ({acc_set}/{seen_imgs})'
                ))

    return x_adv, acc
        