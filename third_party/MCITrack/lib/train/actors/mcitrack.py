from . import BaseActor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy, box_xyxy_to_cxcywh, box_cxcywh_to_xyxy, box_iou
import torch
from lib.utils.heapmap_utils import generate_heatmap

class MCITrackActor(BaseActor):

    def __init__(self, net, objective, loss_weight, settings, cfg):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize
        self.cfg = cfg

    def __call__(self, data):


        out_dict = self.forward_pass(data)


        loss, status = self.compute_losses(out_dict, data)

        return loss, status

    def forward_pass(self, data):
        b = data['search_images'].shape[1]   
        search_list = data['search_images'].view(-1, *data['search_images'].shape[2:]).split(b,dim=0)  
        template_list = data['template_images'].view(-1, *data['template_images'].shape[2:]).split(b,dim=0)
        template_anno_list = data['template_anno'].view(-1, *data['template_anno'].shape[2:]).split(b,dim=0)
        out_list = []
        neck_h_state = [None] * self.cfg.MODEL.NECK.N_LAYERS
        for i in range(len(search_list)):
            search_i_list = [search_list[i]]
            enc_opt = self.net(template_list=template_list, search_list=search_i_list, template_anno_list=template_anno_list, mode='encoder') 
            encoder_out,neck_out,neck_h_state = self.net(enc_opt=enc_opt,neck_h_state=neck_h_state,mode="neck")
            outputs = self.net(feature=neck_out, mode="decoder")
            out_dict = outputs
            out_list.append(out_dict)

        return out_list

    def compute_losses(self, pred_dict, gt_dict, return_status=True):
        total_status = {}
        total_loss = torch.tensor(0., dtype=torch.float).cuda()
        gt_gaussian_maps_list = generate_heatmap(gt_dict['search_anno'], self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.ENCODER.STRIDE) 
        for i in range(len(pred_dict)):

            gt_bbox = gt_dict['search_anno'][i]  
            gt_gaussian_maps = gt_gaussian_maps_list[i].unsqueeze(1)


            pred_boxes = pred_dict[i]['pred_boxes'] 
            if torch.isnan(pred_boxes).any():
                raise ValueError("Network outputs is NAN! Stop Training")
            num_queries = pred_boxes.size(1)
            pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)
            gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0,
                                                                                                               max=1.0)

            try:
                giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  
            except:
                giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()

            l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  

            if 'score_map' in pred_dict[i]:
                location_loss = self.objective['focal'](pred_dict[i]['score_map'], gt_gaussian_maps)
            else:
                location_loss = torch.tensor(0.0, device=l1_loss.device)

            loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight['focal'] * location_loss
            total_loss += loss

            if return_status:

                mean_iou = iou.detach().mean()
                status = {f"{i}frame_Loss/total": loss.item(),
                        f"{i}frame_Loss/giou": giou_loss.item(),
                        f"{i}frame_Loss/l1": l1_loss.item(),
                        f"{i}frame_Loss/location": location_loss.item(),
                        f"{i}frame_IoU": mean_iou.item()}
                total_status.update(status)

        if return_status:
            return total_loss, total_status
        else:
            return total_loss
