from utils import *
import models


def save_checkpoint(state, is_best, path, filename='checkpoint.pth.tar'):
    torch.save(state, "{}/{}".format(path, filename))
    if is_best:
        shutil.copyfile("{}/{}".format(path, filename), "{}/model_best.pth.tar".format(path))

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Model:
    def __init__(self, hps):
        self.hps = hps
        self.net = getattr(models, hps.net)(hps)
        self.optimizer = None
        self.bce_loss = nn.BCELoss()
        self.on_gpu = torch.cuda.is_available()
        if self.on_gpu:
            self.bce_loss = self.bce_loss.cuda()
            self.net = torch.nn.DataParallel(self.net, device_ids=list(range(self.hps.num_gpu))).cuda()


    def init_optimizer(self):
        if self.hps.opt=="sgd": self.optimizer = torch.optim.SGD(self.net.parameters(), self.hps.lr, momentum=0.9,
                                                       weight_decay=self.hps.weight_decay)
        elif self.hps.opt=="adam": self.optimizer = torch.optim.Adam(self.net.parameters(), self.hps.lr,
                                                           weight_decay=self.hps.weight_decay)


    def calc_losses(self, ys, y_preds):
        losses = []
        for cls_idx in range(y_preds.size(1)):
            y, y_pred = ys[:, cls_idx], y_preds[:, cls_idx]
            loss = self.calc_cls_loss(y, y_pred)
            losses.append(loss)
        return losses


    def calc_cls_loss(self, y, y_pred):
        hps = self.hps
        loss = 0.
        if hps.log_loss_weight:
            bce = self.bce_loss(y_pred, y)
            loss += bce * hps.log_loss_weight
    #         print("BCE: {:.3f}".format(bce))
        if hps.dice_loss_weight:
            intersection = (y_pred * y).sum()
            uwi = y_pred.sum() + y.sum()  # without intersection union
            if uwi[0] != 0:
                dice = (1 - intersection / uwi)
                loss += dice * hps.dice_loss_weight
    #             print("Dice: {:.3f}".format(dice))
        loss /= (hps.log_loss_weight + hps.dice_loss_weight)
    #     print("Total weighted: {:.3f}".format(loss))
        return loss


    def process_and_predict(self, img, model):
        pred = model(img)[0, 0]
        pred[pred > 0.5] = 1
        pred[pred != 1] = 0
        return pred


    def validate(self, val_loader, epoch, logger, writer):
        batch_time = AverageMeter()
        losses_meter = AverageMeter()
        #     jaccard_meter = AverageMeter()
        tps, fps, fns = 0, 0, 0
        pics = {}
        for class_ in self.hps.classes:
            pics[class_list[class_]] = []
        # switch to evaluate mode
        classes_string = "_".join([class_list[j] for j in self.hps.classes])
        self.net.eval()

        with torch.no_grad():
            end = time.time()
            for i, sample in enumerate(val_loader):
                img, mask = sample["image"], sample["mask"]
                mask = mask.cuda(non_blocking=True).float()
                h, w = img.size()[2:]
                #             pdb.set_trace()
                y_pred = torch.zeros((1, len(self.hps.classes), h, w), dtype=torch.float).cuda()
                y_pred[:, :, :int(0.5 * h), :int(0.5 * w)] = self.net(img[:, :, :int(0.5 * h), :int(0.5 * w)])[0, 0]
                y_pred[:, :, :int(0.5 * h), int(0.5 * w):] = self.net(img[:, :, :int(0.5 * h), int(0.5 * w):])[0, 0]
                y_pred[:, :, int(0.5 * h):, :int(0.5 * w)] = self.net(img[:, :, int(0.5 * h):, :int(0.5 * w)])[0, 0]
                y_pred[:, :, int(0.5 * h):, int(0.5 * w):] = self.net(img[:, :, int(0.5 * h):, int(0.5 * w):])[0, 0]

                for class_ in range(len(self.hps.classes)):
                    pics[class_list[class_]].append(masktensor2image(mask[:, class_]))
                    # pdb.set_trace()
                    pics[class_list[class_]].append(masktensor2image(y_pred[:, class_]))

                # logger.log(images={'Ground Truth {}'.format(i): mask, 'Prediction {}'.format(i): y_pred}, env=arg_name)
                #             # compute output
                #             y_pred = model(img)
                batch_size = img.size()[0]
                losses = self.calc_losses(mask, y_pred)
                cls_losses = [float(l.item()) for l in losses]
                losses_meter.update(np.mean(cls_losses), batch_size)

                _tp, _fp, _fn = self.mask_tp_fp_fn(y_pred, mask, 0.5)
                tps += _tp.item()
                fps += _fp.item()
                fns += _fn.item()

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                #             if i % arg_print_freq == 0:
                #                 print('Test: [{0}/{1}]\t'
                #                       'Time {batch_time.avg:.3f}\t'
                #                       'Loss {loss.avg:.4f}\t'
                #                       'Jaccard {jaccard.avg:.4f}'.format(
                #                        i, len(val_loader), batch_time=batch_time,
                #                           loss=losses_meter, jaccard=jaccard_meter))
        for class_ in range(len(self.hps.classes)):
            logger.log(image_grid={"GT vs Predictions {}".format(class_list[class_]): pics[class_list[class_]]})
        jaccard_current = 0 if tps == 0 else tps / (tps + fns + fps)
        #     jaccard_meter.update(jaccard_current, batch_size*i)

        print(
            ' * Loss {loss.avg:.3f} Jaccard {jaccard:.3f} (Validation)'.format(loss=losses_meter, jaccard=jaccard_current))
        logger.log(epoch, {'loss_val_{}'.format(classes_string): losses_meter.avg,
                           'jaccard_val_{}'.format(classes_string): jaccard_current})
        writer.add_scalar("val/loss", losses_meter.avg, epoch)
        writer.add_scalar("val/jaccard", jaccard_current, epoch)
        return jaccard_current


    def train_epoch(self, train_loader, epoch, n_iter, total_time, logger, writer):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_meter = AverageMeter()
        jaccard_meter = AverageMeter()
        classes_string = "_".join([class_list[j] for j in self.hps.classes])

        # switch to train mode
        self.net.train()

        end = time.time()
        count = 0
        for i, sample in enumerate(train_loader):
            #         if count>10:
            #             return n_iter, total_time
            img, mask = sample["image"], sample["mask"]
            data_time.update(time.time() - end)
            #         pdb.set_trace()
            #         print(100*mask.sum().float()/(mask.size(0)*mask.size(1)*mask.size(2)*mask.size(3)))
            mask = mask.cuda(non_blocking=True).float()

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            y_pred = self.net(img)
            # pdb.set_trace()
            batch_size = img.size()[0]
            losses = self.calc_losses(mask, y_pred)
            cls_losses = [float(l.item()) for l in losses]
            loss = losses[0]
            for l in losses[1:]:
                loss += l
            losses_meter.update(np.mean(cls_losses), batch_size)
            writer.add_scalar("train/loss", loss.item(), n_iter)
            loss.backward()
            self.optimizer.step()

            _tp, _fp, _fn = self.mask_tp_fp_fn(y_pred, mask, 0.5)
            #         if _tp.item()==0
            jaccard_current = self.jaccard(_tp, _fp, _fn)
            jaccard_meter.update(jaccard_current, batch_size)
            writer.add_scalar("train/jaccard", jaccard_current, n_iter)
            #         logger.log({'loss_trn': losses_meter.avg, 'jaccard_trn': jaccard_current})

            # measure elapsed time
            batch_time.update(time.time() - end)
            total_time += time.time() - end
            end = time.time()

            #         if i!=0 and i % arg_print_freq == 0:
            #             print(' * Epoch: [{0}][{1}/{2}]\t'
            #                   'Time {batch_time.avg:.3f}\t'
            #                   'Data {data_time.avg:.3f}\t'
            #                   'Loss {loss.avg:.4f}\t'
            #                   'Jaccard {jaccard.avg:.4f}'.format(
            #                    epoch, i, len(train_loader), batch_time=batch_time,
            #                    data_time=data_time, loss=losses_meter, jaccard=jaccard_meter))
            n_iter += batch_size
            count += 1
        print(
            'Epoch: [{0}] TotalTime: {total_time:.1f} mins,  BatchTime: {batch_time.avg:.3f},  DataTime: {data_time.avg:.3f},  Loss: {loss.avg:.4f},  Jaccard: {jaccard.avg:.4f}'.format(
                epoch, total_time=total_time / 60, batch_time=batch_time, data_time=data_time, loss=losses_meter,
                jaccard=jaccard_meter))
        logger.log(epoch, {'loss_trn_{}'.format(classes_string): losses_meter.avg,
                           'jaccard_trn_{}'.format(classes_string): jaccard_meter.avg})
        #     print(' * Loss {loss.avg:.3f} Jaccard {jaccard.avg:.3f} (Train)'
        #               .format(loss=losses_meter, jaccard=jaccard_meter))
        return n_iter, total_time


    def mask_tp_fp_fn(self, pred_mask, true_mask, threshold):
        #     pdb.set_trace()
        pred_mask = pred_mask >= threshold
        true_mask = true_mask == 1
        return ((pred_mask & true_mask).sum(),
                (pred_mask & ~true_mask).sum(),
                (~pred_mask & true_mask).sum())

    @staticmethod
    def jaccard(tp, fp, fn):
        if tp.item() == 0: return 0
        return tp.item() / (tp.item() + fn.item() + fp.item())