import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random

try:
    from autopgd_pt import L1_projection
    from other_utils import L1_norm, L2_norm, L0_norm
except ImportError:
    from autoattack.autopgd_base import L1_projection
    from autoattack.other_utils import L1_norm, L2_norm, L0_norm, Logger
    from autoattack.checks import check_zero_gradients
#



def dlr_loss(x, y, reduction='none'):
    x_sorted, ind_sorted = x.sort(dim=1)
    ind = (ind_sorted[:, -1] == y).float()
        
    return -(x[torch.arange(x.shape[0]), y] - x_sorted[:, -2] * ind - \
        x_sorted[:, -1] * (1. - ind)) / (x_sorted[:, -1] - x_sorted[:, -3] + 1e-12)

def dlr_loss_targeted(x, y, y_target):
    x_sorted, ind_sorted = x.sort(dim=1)
    u = torch.arange(x.shape[0])

    return -(x[u, y] - x[u, y_target]) / (x_sorted[:, -1] - .5 * (
        x_sorted[:, -3] + x_sorted[:, -4]) + 1e-12)


def binary_margin_loss(logits, y):

    y_onehot = F.one_hot(y, num_classes=2)
    loss = logits * (1 - y_onehot) - logits * y_onehot
    return loss.sum(-1)


criterion_dict = {'ce': lambda x, y: F.cross_entropy(x, y, reduction='none'),
    'dlr': dlr_loss, 'dlr-targeted': dlr_loss_targeted,
    'binary-margin': binary_margin_loss}

def check_oscillation(x, j, k, y5, k3=0.75):
        t = torch.zeros(x.shape[1]).to(x.device)
        for counter5 in range(k):
          t += (x[j - counter5] > x[j - counter5 - 1]).float()

        return (t <= k * k3 * torch.ones_like(t)).float()


def apgd_train(model, x, y, norm, eps, n_iter=10, use_rs=False, loss='ce',
    verbose=False, is_train=True, early_stop=False, alpha_init=None,
    const_step_size=False, x_init=None, grad_type=None):
    if isinstance(model, nn.Module):
        assert not model.training
    else:
        print('Warning: could not assert if the model is in training mode.')
    device = x.device
    ndims = len(x.shape) - 1
    if grad_type is not None:
        assert norm == 'Linf'
        print(f'Using gradient step: {grad_type}.')
    
    if not use_rs:
        x_adv = x.clone()
    else:
        #raise NotImplemented
        if norm == 'Linf':
            t = (2 * torch.rand_like(x) - 1) * eps
            x_adv = x + t
            x_adv = x_adv.clamp(0., 1.)
        elif norm == 'L2':
            t = torch.randn_like(x)
            t = t / (L2_norm(t, keepdim=True) + 1e-12) * eps
            x_adv = x + t
            x_adv = x_adv.clamp(0., 1.)
        else:
            raise NotImplementedError()
    if x_init is not None:
        x_adv = x_init.clone()
        print('Using custom initialization.')
        print((x_adv - x).abs().max())
    
    x_adv = x_adv.clamp(0., 1.)
    x_best = x_adv.clone()
    x_best_adv = x_adv.clone()
    loss_steps = torch.zeros([n_iter, x.shape[0]], device=device)
    loss_best_steps = torch.zeros([n_iter + 1, x.shape[0]], device=device)
    acc_steps = torch.zeros_like(loss_best_steps)
    
    # set loss
    if isinstance(loss, str):
        criterion_indiv = criterion_dict[loss]
    else:
        criterion_indiv = loss  # To directly pass a function instead of a name.

    # set params
    n_fts = math.prod(x.shape[1:])
    if norm in ['Linf', 'L2']:
        n_iter_2 = max(int(0.22 * n_iter), 1)
        n_iter_min = max(int(0.06 * n_iter), 1)
        size_decr = max(int(0.03 * n_iter), 1)
        k = n_iter_2 + 0
        thr_decr = .75
        alpha = 2.
    elif norm in ['L1']:
        k = max(int(.04 * n_iter), 1)
        init_topk = .05 if is_train else .2
        topk = init_topk * torch.ones([x.shape[0]], device=device)
        sp_old =  n_fts * torch.ones_like(topk)
        adasp_redstep = 1.5
        adasp_minstep = 10.
        alpha = 1.
    if alpha_init is not None:
        alpha = alpha_init
        print(f'Setting alpha={alpha}.')
    
    step_size = alpha * eps * torch.ones([x.shape[0], *[1] * ndims],
        device=device)
    counter3 = 0

    x_adv.requires_grad_()
    #grad = torch.zeros_like(x)
    #for _ in range(self.eot_iter)
    #with torch.enable_grad()
    logits = model(x_adv)
    loss_indiv = criterion_indiv(logits, y)
    loss = loss_indiv.sum()
    #grad += torch.autograd.grad(loss, [x_adv])[0].detach()
    grad = torch.autograd.grad(loss, [x_adv])[0].detach()
    #grad /= float(self.eot_iter)
    grad_best = grad.clone()
    x_adv.detach_()
    loss_indiv.detach_()
    loss.detach_()

    check_zero_gradients(grad)
    
    acc = logits.detach().max(1)[1] == y
    acc_steps[0] = acc + 0
    loss_best = loss_indiv.detach().clone()
    loss_best_last_check = loss_best.clone()
    reduced_last_check = torch.ones_like(loss_best)
    n_reduced = 0
    
    u = torch.arange(x.shape[0], device=device)
    x_adv_old = x_adv.clone().detach()
    
    for i in range(n_iter):
        ### gradient step
        if True: #with torch.no_grad()
            x_adv = x_adv.detach()
            grad2 = x_adv - x_adv_old
            x_adv_old = x_adv.clone()
            loss_curr = loss.detach().mean()
            
            a = 0.75 if i > 0 else 1.0

            if norm == 'Linf':
                if grad_type is None:
                    x_adv_1 = x_adv + step_size * torch.sign(grad)
                elif grad_type == 'L2-norm':
                    x_adv_1 = x_adv + step_size * grad / (L2_norm(grad,
                    keepdim=True) + 1e-12)
                else:
                    raise ValueError(f'Uknown gradient step: {grad_type}.')
                x_adv_1 = torch.clamp(torch.min(torch.max(x_adv_1,
                    x - eps), x + eps), 0.0, 1.0)
                x_adv_1 = torch.clamp(torch.min(torch.max(
                    x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a),
                    x - eps), x + eps), 0.0, 1.0)

            elif norm == 'L2':
                x_adv_1 = x_adv + step_size * grad / (L2_norm(grad,
                    keepdim=True) + 1e-12)
                x_adv_1 = torch.clamp(x + (x_adv_1 - x) / (L2_norm(x_adv_1 - x,
                    keepdim=True) + 1e-12) * torch.min(eps * torch.ones_like(x),
                    L2_norm(x_adv_1 - x, keepdim=True)), 0.0, 1.0)
                x_adv_1 = x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a)
                x_adv_1 = torch.clamp(x + (x_adv_1 - x) / (L2_norm(x_adv_1 - x,
                    keepdim=True) + 1e-12) * torch.min(eps * torch.ones_like(x),
                    L2_norm(x_adv_1 - x, keepdim=True)), 0.0, 1.0)

            elif norm == 'L1':
                grad_topk = grad.abs().view(x.shape[0], -1).sort(-1)[0]
                topk_curr = torch.clamp((1. - topk) * n_fts, min=0, max=n_fts - 1).long()
                grad_topk = grad_topk[u, topk_curr].view(-1, *[1]*(len(x.shape) - 1))
                sparsegrad = grad * (grad.abs() >= grad_topk).float()
                x_adv_1 = x_adv + step_size * sparsegrad.sign() / (
                    sparsegrad.sign().abs().view(x.shape[0], -1).sum(dim=-1).view(
                    -1, 1, 1, 1) + 1e-10)
                
                delta_u = x_adv_1 - x
                delta_p = L1_projection(x, delta_u, eps)
                x_adv_1 = x + delta_u + delta_p
                
            elif norm == 'L0':
                L1normgrad = grad / (grad.abs().view(grad.shape[0], -1).sum(
                    dim=-1, keepdim=True) + 1e-12).view(grad.shape[0], *[1]*(
                    len(grad.shape) - 1))
                x_adv_1 = x_adv + step_size * L1normgrad * n_fts
                x_adv_1 = L0_projection(x_adv_1, x, eps)
                # TODO: add momentum
                
            x_adv = x_adv_1 + 0.

        ### get gradient
        x_adv.requires_grad_()
        #grad = torch.zeros_like(x)
        #for _ in range(self.eot_iter)
        #with torch.enable_grad()
        logits = model(x_adv)
        loss_indiv = criterion_indiv(logits, y)
        loss = loss_indiv.sum()
        
        #grad += torch.autograd.grad(loss, [x_adv])[0].detach()
        if i < n_iter - 1:
            # save one backward pass
            grad = torch.autograd.grad(loss, [x_adv])[0].detach()
            check_zero_gradients(grad)
        #grad /= float(self.eot_iter)
        x_adv.detach_()
        loss_indiv.detach_()
        loss.detach_()
        
        pred = logits.detach().max(1)[1] == y
        acc = torch.min(acc, pred)
        acc_steps[i + 1] = acc + 0
        ind_pred = (pred == 0).nonzero().squeeze()
        x_best_adv[ind_pred] = x_adv[ind_pred] + 0.
        if verbose:
            str_stats = ' - step size: {:.5f} - topk: {:.2f}'.format(
                step_size.mean(), topk.mean() * n_fts) if norm in ['L1'] else ' - step size: {:.5f}'.format(
                step_size.mean())
            str_stats += f' rob-loss: {(loss_indiv * acc).mean():.4f}'
            print('iteration: {} - best loss: {:.6f} curr loss {:.6f} - robust accuracy: {:.2%}{}'.format(
                i, loss_best.sum(), loss_curr, acc.float().mean(), str_stats))
            #print('pert {}'.format((x - x_best_adv).abs().view(x.shape[0], -1).sum(-1).max()))
        
        ### check step size
        if True: #with torch.no_grad()
          y1 = loss_indiv.detach().clone()
          loss_steps[i] = y1 + 0
          ind = (y1 > loss_best).nonzero().squeeze()
          x_best[ind] = x_adv[ind].clone()
          grad_best[ind] = grad[ind].clone()
          loss_best[ind] = y1[ind] + 0
          loss_best_steps[i + 1] = loss_best + 0

          counter3 += 1

          if counter3 == k and not const_step_size:
              if norm in ['Linf', 'L2']:
                  fl_oscillation = check_oscillation(loss_steps, i, k,
                      loss_best, k3=thr_decr)
                  fl_reduce_no_impr = (1. - reduced_last_check) * (
                      loss_best_last_check >= loss_best).float()
                  fl_oscillation = torch.max(fl_oscillation,
                      fl_reduce_no_impr)
                  reduced_last_check = fl_oscillation.clone()
                  loss_best_last_check = loss_best.clone()

                  if fl_oscillation.sum() > 0:
                      ind_fl_osc = (fl_oscillation > 0).nonzero().squeeze()
                      step_size[ind_fl_osc] /= 2.0
                      n_reduced = fl_oscillation.sum()

                      x_adv[ind_fl_osc] = x_best[ind_fl_osc].clone()
                      grad[ind_fl_osc] = grad_best[ind_fl_osc].clone()
                  
                  counter3 = 0
                  k = max(k - size_decr, n_iter_min)
              
              elif norm == 'L1':
                  # adjust sparsity
                  sp_curr = L0_norm(x_best - x)
                  fl_redtopk = (sp_curr / sp_old) < .95
                  topk = sp_curr / n_fts / 1.5
                  step_size[fl_redtopk] = alpha * eps
                  step_size[~fl_redtopk] /= adasp_redstep
                  step_size.clamp_(alpha * eps / adasp_minstep, alpha * eps)
                  sp_old = sp_curr.clone()
              
                  x_adv[fl_redtopk] = x_best[fl_redtopk].clone()
                  grad[fl_redtopk] = grad_best[fl_redtopk].clone()
              
                  counter3 = 0

        if early_stop and acc.sum() == 0:
            break
              
    return x_best, acc, loss_best, x_best_adv


def apgd_largereps(model, x, y, norm='Linf', eps=8. / 255., n_iter=10,
    loss='ce', verbose=False, n_restarts=1, log_path=None, early_stop=False,
    use_rs=False, alpha_init=None, is_train=False, const_step_size=False):
    """Run apgd with the option of restarts."""

    def _project(z, x, norm, eps):

        if norm == 'Linf':
            delta = z - x
            z = x + delta.clamp(-eps, eps)
        else:
            raise NotImplementedError()

        return z.clamp(0., 1.)

    logger = Logger(log_path)
    n_iters = [int(c * n_iter) for c in [.3, .3]]  # .3, .3
    n_iters.append(n_iter - sum(n_iters))
    progr = [2., 1.5, 1]
    epss = [c * eps for c in progr]
    if alpha_init is not None:
        alphas = [alpha_init / c for c in progr]
    else:
        alphas = [None for _ in progr]

    acc = torch.ones([x.shape[0]], device=x.device) # run on all points
    x_adv = x.clone()
    x_init = None

    for inner_it, inner_eps, inner_alpha in zip(n_iters, epss, alphas):
        logger.log(f'Using eps={inner_eps:.5f} for {inner_it} iterations.')

        if x_init is not None:
            x_init = _project(x_init, x, norm, inner_eps)

        #print('Warning: usign best loss points!')
        _, acc, _, x_init = apgd_train(
            model, x, y,
            n_iter=inner_it,
            use_rs=use_rs,
            verbose=verbose,
            loss=loss,
            eps=inner_eps,
            norm=norm,
            #logger=logger,
            early_stop=early_stop,
            x_init=x_init,
            alpha_init=[alpha_init, inner_alpha][-1],
            is_train=is_train,
            const_step_size=const_step_size,
            )

    return _, acc, x_init


if __name__ == '__main__':
    #pass
    from train_new import parse_args
    from data import load_anydataset
    from utils_eval import check_imgs, load_anymodel_datasets, clean_accuracy

    args = parse_args()
    args.training_set = False
    
    x_test, y_test = load_anydataset(args, device='cpu')
    x, y = x_test.cuda(), y_test.cuda()
    model, l_models = load_anymodel_datasets(args)

    assert not model.training

    if args.attack == 'apgd_train':
        #with torch.no_grad()
        x_best, acc, _, x_adv = apgd_train(model, x, y, norm=args.norm,
            eps=args.eps, n_iter=args.n_iter, verbose=True, loss='ce')
        check_imgs(x_adv, x, args.norm)

    elif args.attack == 'apgd_test':
        from autoattack import AutoAttack
        adversary = AutoAttack(model, norm=args.norm, eps=args.eps)
        #adversary.attacks_to_run = ['apgd-ce']
        #adversary.apgd.verbose = True
        with torch.no_grad():
            x_adv = adversary.run_standard_evaluation(x, y, bs=1000)
        check_imgs(x_adv, x, args.norm)



