"""Evaluation Metrics for Semantic Segmentation"""
import torch
import numpy as np

__all__ = ['SegmentationMetric', 'DepthMetric', 'NormalMetric', 'batch_pix_accuracy', 'batch_intersection_union',
           'pixelAccuracy', 'intersectionAndUnion', 'hist_info', 'compute_score']

class RunningMetric(object):
    def __init__(self, metric_type, n_classes =None):
        self._metric_type = metric_type
        if metric_type == 'ACC':
            self.accuracy = 0.0
            self.num_updates = 0.0
        if metric_type == 'L1':
            self.l1 = 0.0
            self.num_updates = 0.0
        if metric_type == 'IOU':
            if n_classes is None:
                print('ERROR: n_classes is needed for IOU')
            self.num_updates = 0.0
            self._n_classes = n_classes
            self.confusion_matrix = np.zeros((n_classes, n_classes))

    def reset(self):
        if self._metric_type == 'ACC':
            self.accuracy = 0.0
            self.num_updates = 0.0
        if self._metric_type == 'L1':
            self.l1 = 0.0
            self.num_updates = 0.0
        if self._metric_type == 'IOU':
            self.num_updates = 0.0
            self.confusion_matrix = np.zeros((self._n_classes, self._n_classes))

    def _fast_hist(self, pred, gt):
        mask = (gt >= 0) & (gt < self._n_classes)
        hist = np.bincount(
            self._n_classes * gt[mask].astype(int) +
            pred[mask], minlength=self._n_classes**2).reshape(self._n_classes, self._n_classes)
        return hist

    def update(self, pred, gt):
        if self._metric_type == 'ACC':
            predictions = pred.data.max(1, keepdim=True)[1]
            self.accuracy += (predictions.eq(gt.data.view_as(predictions)).cpu().sum()) 
            self.num_updates += predictions.shape[0]
    
        if self._metric_type == 'L1':
            _gt = gt.data.cpu().numpy()
            _pred = pred.data.cpu().numpy()
            gti = _gt.astype(np.int32)
            mask = gti!=250
            if np.sum(mask) < 1:
                return
            self.l1 += np.sum( np.abs(gti[mask] - _pred.astype(np.int32)[mask]) ) 
            self.num_updates += np.sum(mask)

        if self._metric_type == 'IOU':
            _pred = pred.data.max(1)[1].cpu().numpy()
            _gt = gt.data.cpu().numpy()
            for lt, lp in zip(_pred, _gt):
                self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten())
        
    def get(self):
        if self._metric_type == 'ACC':
            return self.accuracy/self.num_updates,0
        if self._metric_type == 'L1':
            return {'l1': self.l1/self.num_updates}
        if self._metric_type == 'IOU':
            acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
            acc_cls = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum(axis=1)
            acc_cls = np.nanmean(acc_cls)
            iou = np.diag(self.confusion_matrix) / (self.confusion_matrix.sum(axis=1) + self.confusion_matrix.sum(axis=0) - np.diag(self.confusion_matrix)) 
            mean_iou = np.nanmean(iou)
            return {'micro_acc': acc, 'macro_acc':acc_cls, 'mIOU': mean_iou}

class NormalMetric(object):
    """Computes pixAcc and mIoU metric scores
    """

    def __init__(self, nclass):
        super(NormalMetric, self).__init__()
        self.nclass = nclass
        self.reset()

    def update(self, preds, labels):
        """Updates the internal evaluation result.

        Parameters
        ----------
        labels : 'NumpyArray' or list of `NumpyArray`
            The labels of the data.
        preds : 'NumpyArray' or list of `NumpyArray`
            Predicted values.
        """

        def evaluate_worker(self, pred, label):
            binary_mask = (torch.sum(label, dim=1) != 0)
            error = torch.acos(torch.clamp(torch.sum(pred * label, 1).masked_select(binary_mask), -1, 1)).detach().cpu().numpy()
            error = np.degrees(error)
            self.mean_ad += np.mean(error) #angle distance
            self.median_ad += np.median(error)
            self.t_11 += np.mean(error < 11.25)
            self.t_22 += np.mean(error < 22.5)
            self.t_30 += np.mean(error < 30)
            self.num += 1
            # print(self.num,self.abs_err, self.rel_err)

        if isinstance(preds, torch.Tensor):
            evaluate_worker(self, preds, labels)
        elif isinstance(preds, (list, tuple)):
            for (pred, label) in zip(preds, labels):
                evaluate_worker(self, pred, label)

    def get(self):
        #Gets the current evaluation result.
        mean_ad = 1.0*self.mean_ad/(2.220446049250313e-16+self.num)
        median_ad = 1.0*self.median_ad/(2.220446049250313e-16+self.num)
        t_11 = 1.0*self.t_11/(2.220446049250313e-16+self.num)
        t_22 = 1.0*self.t_22/(2.220446049250313e-16+self.num)
        t_30 = 1.0*self.t_30/(2.220446049250313e-16+self.num)

        return mean_ad, median_ad, t_11, t_22, t_30

    def reset(self):
        """Resets the internal evaluation result to initial state."""
        self.mean_ad = 0
        self.median_ad = 0
        self.t_11 = 0
        self.t_22 = 0
        self.t_30 = 0
        self.num = 0

class DepthMetric(object):
    """Computes pixAcc and mIoU metric scores
    """

    def __init__(self, nclass, device):
        super(DepthMetric, self).__init__()
        self.nclass = nclass
        self.device = device
        self.reset()

    def update(self, preds, labels):
        """Updates the internal evaluation result.

        Parameters
        ----------
        labels : 'NumpyArray' or list of `NumpyArray`
            The labels of the data.
        preds : 'NumpyArray' or list of `NumpyArray`
            Predicted values.
        """

        def evaluate_worker(self, pred, label):
            binary_mask = (torch.sum(label, dim=1) != 0).unsqueeze(1).to(self.device)
            x_pred_true = pred.masked_select(binary_mask)
            x_output_true = label.masked_select(binary_mask)
            # print(pred, x_pred_true, x_output_true)
            abs_err = torch.abs(x_pred_true - x_output_true)
            rel_err = torch.abs(x_pred_true - x_output_true) / x_output_true
            self.abs_err += (torch.sum(abs_err) / torch.nonzero(binary_mask, as_tuple=False).size(0))
            self.rel_err += (torch.sum(rel_err) / torch.nonzero(binary_mask, as_tuple=False).size(0))
            self.num += 1
            # print(self.num,self.abs_err, self.rel_err)

        if isinstance(preds, torch.Tensor):
            evaluate_worker(self, preds, labels)
        elif isinstance(preds, (list, tuple)):
            for (pred, label) in zip(preds, labels):
                evaluate_worker(self, pred, label)

    def get(self):
        #Gets the current evaluation result.
        abs_err = 1.0*self.abs_err/(2.220446049250313e-16+self.num)
        rel_err = 1.0*self.rel_err/(2.220446049250313e-16+self.num)

        return abs_err, rel_err

    def reset(self):
        """Resets the internal evaluation result to initial state."""
        self.abs_err = 0
        self.rel_err = 0
        self.num = 0

class SegmentationMetric(object):
    """Computes pixAcc and mIoU metric scores
    """

    def __init__(self, nclass):
        super(SegmentationMetric, self).__init__()
        self.nclass = nclass
        self.reset()

    def update(self, preds, labels):
        """Updates the internal evaluation result.

        Parameters
        ----------
        labels : 'NumpyArray' or list of `NumpyArray`
            The labels of the data.
        preds : 'NumpyArray' or list of `NumpyArray`
            Predicted values.
        """

        def evaluate_worker(self, pred, label):
            correct, labeled = batch_pix_accuracy(pred, label)
            inter, union = batch_intersection_union(pred, label, self.nclass)

            self.total_correct += correct
            self.total_label += labeled
            if self.total_inter.device != inter.device:
                self.total_inter = self.total_inter.to(inter.device)
                self.total_union = self.total_union.to(union.device)
            self.total_inter += inter
            self.total_union += union

        # def evaluate_worker(self, pred, label):
        #     correct, labeled = batch_pix_accuracy(pred, label)
        #     inter, union = batch_intersection_union(pred, label, self.nclass)

        #     self.total_correct += correct
        #     self.total_label += labeled
        #     if self.total_inter.device != inter.device:
        #         self.total_inter = self.total_inter.to(inter.device)
        #         self.total_union = self.total_union.to(union.device)
        #     iou = 1.0 * inter / (2.220446049250313e-16 + union)
        #     # print(iou.shape) #torch.size([7])
        #     self.total_miou += iou.mean().item()
        #     self.num += 1

        if isinstance(preds, torch.Tensor):
            evaluate_worker(self, preds, labels)
        elif isinstance(preds, (list, tuple)):
            for (pred, label) in zip(preds, labels):
                evaluate_worker(self, pred, label)

    def get(self):
        """Gets the current evaluation result.

        Returns
        -------
        metrics : tuple of float
            pixAcc and mIoU 
        """
        pixAcc = 1.0 * self.total_correct / (2.220446049250313e-16 + self.total_label)  # remove np.spacing(1)
        IoU = 1.0 * self.total_inter / (2.220446049250313e-16 + self.total_union) #size: [nclass,]
        mIoU = IoU.mean().item()
        # print(self.total_inter.mean().item())
        # mIoU = self.total_miou / self.num
        return pixAcc, mIoU

    def reset(self):
        """Resets the internal evaluation result to initial state."""
        self.total_inter = torch.zeros(self.nclass)
        self.total_union = torch.zeros(self.nclass)
        self.total_correct = 0
        self.total_label = 0
        self.total_miou = 0
        self.num = 0

# class ConfMatrix(object):
#     def __init__(self, num_classes):
#         self.num_classes = num_classes
#         self.mat = None

#     def update(self, pred, target):
#         n = self.num_classes
#         if self.mat is None:
#             self.mat = torch.zeros((n, n), dtype=torch.int64, device=pred.device)
#         with torch.no_grad():
#             k = (target >= 0) & (target < n)
#             inds = n * target[k].to(torch.int64) + pred[k]
#             self.mat += torch.bincount(inds, minlength=n ** 2).reshape(n, n)

#     def get_metrics(self):
#         h = self.mat.float()
#         acc = torch.diag(h).sum() / h.sum()
#         iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
#         return torch.mean(iu), acc
#     def reset(self):
#         """Resets the internal evaluation result to initial state."""
#         self.mat = None
# # pytorch version
def batch_pix_accuracy(output, target):
    """PixAcc"""
    # inputs are numpy array, output 4D, target 3D
    predict = torch.argmax(output.long(), 1) + 1
    target = target.long() + 1
    target = target.squeeze(1)

    pixel_labeled = torch.sum(target > 0).item()
    pixel_correct = torch.sum((predict == target) * (target > 0)).item()

    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled

def batch_intersection_union(output, target, nclass):
    """mIoU"""
    # inputs are numpy array, output 4D, target 3D
    mini = 1
    maxi = nclass
    nbins = nclass
    predict = torch.argmax(output, 1) + 1
    target = target.float() + 1
    target = target.squeeze(1)

    predict = predict.float() * (target > 0).float() #此方式忽略了0像素点，因此最终的z准确率计算结果会偏高。target中也是有很多0像素点的，intersection也是如此，但torch.histc()最小值从1开始
    intersection = predict * (predict == target).float()
    # areas of intersection and union
    # element 0 in intersection occur the main difference from np.bincount. set boundary to -1 is necessary.
    area_inter = torch.histc(intersection.cpu(), bins=nbins, min=mini, max=maxi) #the number of pixel for one class(from 1 to nclass), output:[n1,...,n13],n1:number of pixels for class 1
    area_pred = torch.histc(predict.cpu(), bins=nbins, min=mini, max=maxi)
    area_lab = torch.histc(target.cpu(), bins=nbins, min=mini, max=maxi)
    area_union = area_pred + area_lab - area_inter
    assert torch.sum(area_inter > area_union).item() == 0, "Intersection area should be smaller than Union area"
    return area_inter.float(), area_union.float()


def pixelAccuracy(imPred, imLab):
    """
    This function takes the prediction and label of a single image, returns pixel-wise accuracy
    To compute over many images do:
    for i = range(Nimages):
         (pixel_accuracy[i], pixel_correct[i], pixel_labeled[i]) = \
            pixelAccuracy(imPred[i], imLab[i])
    mean_pixel_accuracy = 1.0 * np.sum(pixel_correct) / (np.spacing(1) + np.sum(pixel_labeled))
    """
    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    pixel_labeled = np.sum(imLab >= 0)
    pixel_correct = np.sum((imPred == imLab) * (imLab >= 0))
    pixel_accuracy = 1.0 * pixel_correct / pixel_labeled
    return (pixel_accuracy, pixel_correct, pixel_labeled)


def intersectionAndUnion(imPred, imLab, numClass):
    """
    This function takes the prediction and label of a single image,
    returns intersection and union areas for each class
    To compute over many images do:
    for i in range(Nimages):
        (area_intersection[:,i], area_union[:,i]) = intersectionAndUnion(imPred[i], imLab[i])
    IoU = 1.0 * np.sum(area_intersection, axis=1) / np.sum(np.spacing(1)+area_union, axis=1)
    """
    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    imPred = imPred * (imLab >= 0)

    # Compute area intersection:
    intersection = imPred * (imPred == imLab)
    (area_intersection, _) = np.histogram(intersection, bins=numClass, range=(1, numClass))

    # Compute area union:
    (area_pred, _) = np.histogram(imPred, bins=numClass, range=(1, numClass))
    (area_lab, _) = np.histogram(imLab, bins=numClass, range=(1, numClass))
    area_union = area_pred + area_lab - area_intersection
    return (area_intersection, area_union)


def hist_info(pred, label, num_cls):
    assert pred.shape == label.shape
    k = (label >= 0) & (label < num_cls)
    labeled = np.sum(k)
    correct = np.sum((pred[k] == label[k]))

    return np.bincount(num_cls * label[k].astype(int) + pred[k], minlength=num_cls ** 2).reshape(num_cls,
                                                                                                 num_cls), labeled, correct


def compute_score(hist, correct, labeled):
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    mean_IU = np.nanmean(iu)
    mean_IU_no_back = np.nanmean(iu[1:])
    freq = hist.sum(1) / hist.sum()
    freq_IU = (iu[freq > 0] * freq[freq > 0]).sum()
    mean_pixel_acc = correct / labeled

    return iu, mean_IU, mean_IU_no_back, mean_pixel_acc


# ```
# @article{zhang2021prototypical,
#     title={Prototypical Pseudo Label Denoising and Target Structure Learning for Domain Adaptive Semantic Segmentation},
#     author={Zhang, Pan and Zhang, Bo and Zhang, Ting and Chen, Dong and Wang, Yong and Wen, Fang},
#     journal={arXiv preprint arXiv:2101.10979},
#     year={2021}
# }
# ```
# class SegmentationMetric(object):
#     def __init__(self, n_classes):
#         self.n_classes = n_classes
#         self.confusion_matrix = np.zeros((n_classes, n_classes))

#     def _fast_hist(self, label_true, label_pred, n_class):
#         mask = (label_true >= 0) & (label_true < n_class)
#         hist = np.bincount(
#             n_class * label_true[mask].astype(int) + label_pred[mask],
#             minlength=n_class ** 2,
#         ).reshape(n_class, n_class)
#         return hist

#     def update(self, label_trues, label_preds):
#         for lt, lp in zip(label_trues, label_preds):
#             self.confusion_matrix += self._fast_hist(
#                 lt.flatten(), lp.flatten(), self.n_classes
#             )

#     def get(self):
#         """Returns accuracy score evaluation result.
#             - overall accuracy
#             - mean accuracy
#             - mean IU
#             - fwavacc
#         """
#         hist = self.confusion_matrix
#         acc = np.diag(hist).sum() / hist.sum()
#         acc_cls = np.diag(hist) / hist.sum(axis=1)
#         acc_cls = np.nanmean(acc_cls)
#         iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
#         mean_iu = np.nanmean(iu)
#         freq = hist.sum(axis=1) / hist.sum()
#         fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
#         cls_iu = dict(zip(range(self.n_classes), iu))

#         return acc_cls, mean_iu
#         # return (
#         #     {
#         #         "Overall Acc: \t": acc,
#         #         "Mean Acc : \t": acc_cls,
#         #         "FreqW Acc : \t": fwavacc,
#         #         "Mean IoU : \t": mean_iu,
#         #     },
#         #     cls_iu,
#         # )

#     def reset(self):
#         self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))