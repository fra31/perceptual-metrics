import torch
import torch.nn.functional as F
#from autoattack import AutoAttack
from autoattack.other_utils import Logger

import autopgd
import utils_perceptual_eval


# Tools to attack 2AFC task.

def binary_margin_loss(logits, y):

    y_onehot = F.one_hot(y, num_classes=2)
    loss = logits * (1 - y_onehot) - logits * y_onehot
    return loss.sum(-1)


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
            
        
        if attack_name == 'apgd':
            out = autopgd.apgd_train(
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
                kwargs['use_rs'] = True
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
        