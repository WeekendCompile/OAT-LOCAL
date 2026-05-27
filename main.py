import os
import json
import torch
import torchvision
import torch.nn.parallel
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import opts_egtea as opts
import time
import h5py
from tqdm import tqdm
from iou_utils import *
from eval import evaluation_detection
from tensorboardX import SummaryWriter
from dataset import VideoDataSet
from models import MYNET, SuppressNet
from loss_func import cls_loss_func, regress_loss_func

def train_one_epoch(opt, model, train_dataset, optimizer, warmup=False):
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=opt['batch_size'], shuffle=True,
                                                num_workers=0, pin_memory=True,drop_last=False)      
    epoch_cost = 0
    epoch_cost_cls = 0
    epoch_cost_reg = 0
    
    total_iter = len(train_dataset)//opt['batch_size']
    
    progress_bar = tqdm(train_loader, desc="Train", unit="batch", leave=False)
    for n_iter,(input_data,cls_label,reg_label,_) in enumerate(progress_bar):
        if warmup:
            for g in optimizer.param_groups:
                g['lr'] = n_iter * (opt['lr']) / total_iter
        
        act_cls, act_reg = model(input_data.float().cuda())
        
        cost_reg = 0
        cost_cls = 0
        
        loss = cls_loss_func(cls_label,act_cls)
        cost_cls = loss
            
        epoch_cost_cls += cost_cls.detach().cpu().item()
               
        loss = regress_loss_func(reg_label,act_reg)
        cost_reg = loss  
        epoch_cost_reg += cost_reg.detach().cpu().item()
        
        cost= opt['alpha']*cost_cls +opt['beta']*cost_reg    
                
        epoch_cost += cost.detach().cpu().item()
        
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        progress_bar.set_postfix(loss=cost.detach().cpu().item())
                
    return n_iter, epoch_cost, epoch_cost_cls, epoch_cost_reg

def eval_one_epoch(opt, model, test_dataset):
    cls_loss, reg_loss, tot_loss, output_cls, output_reg, labels_cls, labels_reg, working_time, total_frames = eval_frame(opt, model,test_dataset)
        
    result_dict = eval_map_nms(opt,test_dataset, output_cls, output_reg, labels_cls, labels_reg)
    output_dict={"version":"VERSION 1.3","results":result_dict,"external_data":{}}
    outfile=open(opt["result_file"].format(opt['exp']),"w")
    json.dump(output_dict,outfile, indent=2)
    outfile.close()
    
    IoUmAP = evaluation_detection(opt, verbose=False)
    IoUmAP_5=sum(IoUmAP[0:])/len(IoUmAP[0:])
    return cls_loss, reg_loss, tot_loss, IoUmAP_5

    
def train(opt): 
    writer = SummaryWriter()
    model = MYNET(opt).cuda()
    
    optimizer = optim.Adam( model.parameters(),lr=opt["lr"],weight_decay = opt["weight_decay"])      
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size = opt["lr_step"])
    
    train_dataset = VideoDataSet(opt,subset="train")      
    test_dataset = VideoDataSet(opt,subset=opt['inference_subset'])
    
    warmup=False
    
    for n_epoch in range(opt['epoch']):   
        if n_epoch >=1:
            warmup=False
        
        n_iter, epoch_cost, epoch_cost_cls, epoch_cost_reg = train_one_epoch(opt, model, train_dataset, optimizer, warmup)
            
        writer.add_scalars('data/cost', {'train': epoch_cost/(n_iter+1)}, n_epoch)
        print("training loss(epoch %d): %.03f, cls - %f, reg - %f, lr - %f"%(n_epoch,
                                                                            epoch_cost/(n_iter+1),
                                                                            epoch_cost_cls/(n_iter+1),
                                                                            epoch_cost_reg/(n_iter+1),
                                                                            optimizer.param_groups[0]["lr"]) )
        
        scheduler.step()
        model.eval()
        
        cls_loss, reg_loss, tot_loss, IoUmAP_5 = eval_one_epoch(opt, model,test_dataset)
        
        writer.add_scalars('data/mAP', {'test': IoUmAP_5}, n_epoch)
        print("testing loss(epoch %d): %.03f, cls - %f, reg - %f, mAP Avg - %f"%(n_epoch,tot_loss, cls_loss, reg_loss, IoUmAP_5))
                    
        state = {'epoch': n_epoch + 1,
                    'state_dict': model.state_dict()}
        torch.save(state, opt["checkpoint_path"]+"/checkpoint.pth.tar" )
        if IoUmAP_5 > model.best_map:
            model.best_map = IoUmAP_5
            torch.save(state, opt["checkpoint_path"]+"/ckp_best.pth.tar" )
            
        model.train()
                
    writer.close()
    return model.best_map

def eval_frame(opt, model, dataset):
    test_loader = torch.utils.data.DataLoader(dataset,
                                                batch_size=opt['batch_size'], shuffle=False,
                                                num_workers=0, pin_memory=True,drop_last=False)
    
    labels_cls={}
    labels_reg={}
    output_cls={}
    output_reg={}                                      
    for video_name in dataset.video_list:
        labels_cls[video_name]=[]
        labels_reg[video_name]=[]
        output_cls[video_name]=[]
        output_reg[video_name]=[]
        
    start_time = time.time()
    total_frames =0  
    epoch_cost = 0
    epoch_cost_cls = 0
    epoch_cost_reg = 0   
    
    progress_bar = tqdm(test_loader, desc="Test", unit="batch", leave=False)
    for n_iter,(input_data,cls_label,reg_label,_) in enumerate(progress_bar):
        act_cls, act_reg = model(input_data.float().cuda())
        
        cost_reg = 0
        cost_cls = 0
        
        loss = cls_loss_func(cls_label,act_cls)
        cost_cls = loss
            
        epoch_cost_cls += cost_cls.detach().cpu().item()
               
        loss = regress_loss_func(reg_label,act_reg)
        cost_reg = loss  
        epoch_cost_reg += cost_reg.detach().cpu().item()
        
        cost= opt['alpha']*cost_cls +opt['beta']*cost_reg    
                
        epoch_cost += cost.detach().cpu().item()
        
        act_cls = torch.softmax(act_cls, dim=-1)
        
        total_frames+=input_data.size(0)
        
        for b in range(0,input_data.size(0)):
            video_name, st, ed, data_idx = dataset.inputs[n_iter*opt['batch_size']+b]
            output_cls[video_name]+=[act_cls[b,:].detach().cpu().numpy()]
            output_reg[video_name]+=[act_reg[b,:].detach().cpu().numpy()]
            labels_cls[video_name]+=[cls_label[b,:].numpy()]
            labels_reg[video_name]+=[reg_label[b,:].numpy()]
        progress_bar.set_postfix(loss=cost.detach().cpu().item())
        
    end_time = time.time()
    working_time = end_time-start_time
    
    for video_name in dataset.video_list:
        labels_cls[video_name]=np.stack(labels_cls[video_name], axis=0)
        labels_reg[video_name]=np.stack(labels_reg[video_name], axis=0)
        output_cls[video_name]=np.stack(output_cls[video_name], axis=0)
        output_reg[video_name]=np.stack(output_reg[video_name], axis=0)
    
    cls_loss=epoch_cost_cls/n_iter
    reg_loss=epoch_cost_reg/n_iter
    tot_loss=epoch_cost/n_iter
     
    return cls_loss, reg_loss, tot_loss, output_cls, output_reg, labels_cls, labels_reg, working_time, total_frames


def eval_map_nms(opt, dataset, output_cls, output_reg, labels_cls, labels_reg):
    result_dict={}
    proposal_dict=[]
    
    num_class = opt["num_of_class"]
    unit_size = opt['segment_size']
    threshold=opt['threshold']
    anchors=opt['anchors']
                                             
    for video_name in dataset.video_list:
        duration = dataset.video_len[video_name]
        video_time = float(dataset.video_dict[video_name]["duration"])
        frame_to_time = 100.0*video_time / duration
         
        for idx in range(0,duration):
            cls_anc = output_cls[video_name][idx]
            reg_anc = output_reg[video_name][idx]
            
            proposal_anc_dict=[]
            for anc_idx in range(0,len(anchors)):
                cls = np.argwhere(cls_anc[anc_idx][:-1]>opt['threshold']).reshape(-1)
                
                if len(cls) == 0:
                    continue
                    
                ed= idx + anchors[anc_idx] * reg_anc[anc_idx][0]
                length = anchors[anc_idx]* np.exp(reg_anc[anc_idx][1])
                st= ed-length
                
                for cidx in range(0,len(cls)):
                    label=cls[cidx]
                    tmp_dict={}
                    tmp_dict["segment"] = [st*frame_to_time/100.0, ed*frame_to_time/100.0]
                    tmp_dict["score"]= cls_anc[anc_idx][label]*1.0
                    tmp_dict["label"]=dataset.label_name[label]
                    tmp_dict["gentime"]= idx*frame_to_time/100.0
                    proposal_anc_dict.append(tmp_dict)
                
            proposal_dict+=proposal_anc_dict
        
        proposal_dict=non_max_suppression(proposal_dict, overlapThresh=opt['soft_nms'])
                    
        result_dict[video_name]=proposal_dict
        proposal_dict=[]
        
    return result_dict


def eval_map_supnet(opt, dataset, output_cls, output_reg, labels_cls, labels_reg):
    model = SuppressNet(opt).cuda()
    checkpoint = torch.load(opt["checkpoint_path"]+"/ckp_best_suppress.pth.tar")
    base_dict=checkpoint['state_dict']
    model.load_state_dict(base_dict)
    model.eval()
    
    result_dict={}
    proposal_dict=[]
    
    num_class = opt["num_of_class"]
    unit_size = opt['segment_size']
    threshold=opt['threshold']
    anchors=opt['anchors']
                                             
    for video_name in dataset.video_list:
        duration = dataset.video_len[video_name]
        video_time = float(dataset.video_dict[video_name]["duration"])
        frame_to_time = 100.0*video_time / duration
        conf_queue = torch.zeros((unit_size,num_class-1)) 
        
        for idx in range(0,duration):
            cls_anc = output_cls[video_name][idx]
            reg_anc = output_reg[video_name][idx]
            
            proposal_anc_dict=[]
            for anc_idx in range(0,len(anchors)):
                cls = np.argwhere(cls_anc[anc_idx][:-1]>opt['threshold']).reshape(-1)
                
                if len(cls) == 0:
                    continue
                    
                ed= idx + anchors[anc_idx] * reg_anc[anc_idx][0]
                length = anchors[anc_idx]* np.exp(reg_anc[anc_idx][1])
                st= ed-length
                
                for cidx in range(0,len(cls)):
                    label=cls[cidx]
                    tmp_dict={}
                    tmp_dict["segment"] = [st*frame_to_time/100.0, ed*frame_to_time/100.0]
                    tmp_dict["score"]= cls_anc[anc_idx][label]*1.0
                    tmp_dict["label"]=dataset.label_name[label]
                    tmp_dict["gentime"]= idx*frame_to_time/100.0
                    proposal_anc_dict.append(tmp_dict)
                          
            proposal_anc_dict = non_max_suppression(proposal_anc_dict, overlapThresh=opt['soft_nms'])  
                
            conf_queue[:-1,:]=conf_queue[1:,:].clone()
            conf_queue[-1,:]=0
            for proposal in proposal_anc_dict:
                cls_idx = dataset.label_name.index(proposal['label'])
                conf_queue[-1,cls_idx]=proposal["score"]
            
            minput = conf_queue.unsqueeze(0)
            suppress_conf = model(minput.cuda())
            suppress_conf=suppress_conf.squeeze(0).detach().cpu().numpy()
            
            for cls in range(0,num_class-1):
                if suppress_conf[cls] > opt['sup_threshold']:
                    for proposal in proposal_anc_dict:
                        if proposal['label'] == dataset.label_name[cls]:
                            if check_overlap_proposal(proposal_dict, proposal, overlapThresh=opt['soft_nms']) is None:
                                proposal_dict.append(proposal)
            
        result_dict[video_name]=proposal_dict
        proposal_dict=[]
        
    return result_dict

 
def test_frame(opt): 
    model = MYNET(opt).cuda()
    checkpoint = torch.load(opt["checkpoint_path"]+"/ckp_best.pth.tar")
    base_dict=checkpoint['state_dict']
    model.load_state_dict(base_dict)
    model.eval()
    
    dataset = VideoDataSet(opt,subset=opt['inference_subset'])    
    outfile = h5py.File(opt['frame_result_file'], 'w')
    
    cls_loss, reg_loss, tot_loss, output_cls, output_reg, labels_cls, labels_reg, working_time, total_frames = eval_frame(opt, model,dataset)
    
    print("testing loss: %f, cls_loss: %f, reg_loss: %f"%(tot_loss, cls_loss, reg_loss ))
    
    for video_name in dataset.video_list:
        o_cls=output_cls[video_name]
        o_reg=output_reg[video_name]
        l_cls=labels_cls[video_name]
        l_reg=labels_reg[video_name]
        
        dset_predcls = outfile.create_dataset(video_name+'/pred_cls', o_cls.shape, maxshape=o_cls.shape, chunks=True, dtype=np.float32)
        dset_predcls[:,:] = o_cls[:,:]  
        dset_predreg = outfile.create_dataset(video_name+'/pred_reg', o_reg.shape, maxshape=o_reg.shape, chunks=True, dtype=np.float32)
        dset_predreg[:,:] = o_reg[:,:]   
        dset_labelcls = outfile.create_dataset(video_name+'/label_cls', l_cls.shape, maxshape=l_cls.shape, chunks=True, dtype=np.float32)
        dset_labelcls[:,:] = l_cls[:,:]   
        dset_labelreg = outfile.create_dataset(video_name+'/label_reg', l_reg.shape, maxshape=l_reg.shape, chunks=True, dtype=np.float32)
        dset_labelreg[:,:] = l_reg[:,:]   
    outfile.close()
                    
    print("working time : {}s, {}fps, {} frames".format(working_time, total_frames/working_time, total_frames))
    

def test(opt): 
    model = MYNET(opt).cuda()
    checkpoint = torch.load(opt["checkpoint_path"]+"/ckp_best.pth.tar")
    base_dict=checkpoint['state_dict']
    model.load_state_dict(base_dict)
    model.eval()
    
    dataset = VideoDataSet(opt,subset=opt['inference_subset'])
    
    cls_loss, reg_loss, tot_loss, output_cls, output_reg, labels_cls, labels_reg, working_time, total_frames = eval_frame(opt, model,dataset)
    
    if opt["pptype"]=="nms":
        result_dict = eval_map_nms(opt,dataset, output_cls, output_reg, labels_cls, labels_reg)
    if opt["pptype"]=="net":
        result_dict = eval_map_supnet(opt,dataset, output_cls, output_reg, labels_cls, labels_reg)
    output_dict={"version":"VERSION 1.3","results":result_dict,"external_data":{}}
    outfile=open(opt["result_file"].format(opt['exp']),"w")
    json.dump(output_dict,outfile, indent=2)
    outfile.close()
    
    mAP = evaluation_detection(opt)


def test_online(opt): 
    model = MYNET(opt).cuda()
    checkpoint = torch.load(opt["checkpoint_path"]+"/ckp_best.pth.tar")
    base_dict=checkpoint['state_dict']
    model.load_state_dict(base_dict)
    model.eval()
    
    sup_model = SuppressNet(opt).cuda()
    checkpoint = torch.load(opt["checkpoint_path"]+"/ckp_best_suppress.pth.tar")
    base_dict=checkpoint['state_dict']
    sup_model.load_state_dict(base_dict)
    sup_model.eval()
    
    dataset = VideoDataSet(opt,subset=opt['inference_subset'])
    test_loader = torch.utils.data.DataLoader(dataset,
                                                batch_size=1, shuffle=False,
                                                num_workers=0, pin_memory=True,drop_last=False)
    
    result_dict={}
    proposal_dict=[]
    
    
    num_class = opt["num_of_class"]
    unit_size = opt['segment_size']
    threshold=opt['threshold']
    anchors=opt['anchors']
    
    start_time = time.time()
    total_frames =0 
    
    
    video_pbar = tqdm(dataset.video_list, desc="Videos", unit="vid")
    for video_name in video_pbar:
        input_queue = torch.zeros((unit_size,opt['feat_dim'])) 
        sup_queue = torch.zeros(((unit_size,num_class-1)))
    
        duration = dataset.video_len[video_name]
        video_time = float(dataset.video_dict[video_name]["duration"])
        frame_to_time = 100.0*video_time / duration
        
        frame_pbar = tqdm(range(0,duration), desc=video_name[:20], unit="f", leave=False)
        for idx in frame_pbar:
            total_frames+=1
            input_queue[:-1,:]=input_queue[1:,:].clone()
            input_queue[-1:,:]=dataset._get_base_data(video_name,idx,idx+1)
            
            minput = input_queue.unsqueeze(0)
            act_cls, act_reg = model(minput.cuda())
            act_cls = torch.softmax(act_cls, dim=-1)
            
            cls_anc = act_cls.squeeze(0).detach().cpu().numpy()
            reg_anc = act_reg.squeeze(0).detach().cpu().numpy()
            
            proposal_anc_dict=[]
            for anc_idx in range(0,len(anchors)):
                cls = np.argwhere(cls_anc[anc_idx][:-1]>opt['threshold']).reshape(-1)
                
                if len(cls) == 0:
                    continue
                    
                ed= idx + anchors[anc_idx] * reg_anc[anc_idx][0]
                length = anchors[anc_idx]* np.exp(reg_anc[anc_idx][1])
                st= ed-length
                
                for cidx in range(0,len(cls)):
                    label=cls[cidx]
                    tmp_dict={}
                    tmp_dict["segment"] = [st*frame_to_time/100.0, ed*frame_to_time/100.0]
                    tmp_dict["score"]= cls_anc[anc_idx][label]*1.0
                    tmp_dict["label"]=dataset.label_name[label]
                    tmp_dict["gentime"]= idx*frame_to_time/100.0
                    proposal_anc_dict.append(tmp_dict)
                          
            proposal_anc_dict = non_max_suppression(proposal_anc_dict, overlapThresh=opt['soft_nms'])  
                
            sup_queue[:-1,:]=sup_queue[1:,:].clone()
            sup_queue[-1,:]=0
            for proposal in proposal_anc_dict:
                cls_idx = dataset.label_name.index(proposal['label'])
                sup_queue[-1,cls_idx]=proposal["score"]
            
            minput = sup_queue.unsqueeze(0)
            suppress_conf = sup_model(minput.cuda())
            suppress_conf=suppress_conf.squeeze(0).detach().cpu().numpy()
            
            for cls in range(0,num_class-1):
                if suppress_conf[cls] > opt['sup_threshold']:
                    for proposal in proposal_anc_dict:
                        if proposal['label'] == dataset.label_name[cls]:
                            if check_overlap_proposal(proposal_dict, proposal, overlapThresh=opt['soft_nms']) is None:
                                proposal_dict.append(proposal)
            
        result_dict[video_name]=proposal_dict
        proposal_dict=[]
        elapsed = time.time() - start_time
        running_fps = total_frames / elapsed if elapsed > 0 else 0.0
        video_pbar.set_postfix(frames=total_frames, fps="{:.1f}".format(running_fps))
    
    end_time = time.time()
    working_time = end_time-start_time
    avg_fps = total_frames / working_time if working_time > 0 else 0.0
    per_frame_ms = (working_time / total_frames * 1000.0) if total_frames > 0 else 0.0

    print("\n=============== Streaming Inference Speed (Online, batch=1) ===============")
    print("Total frames processed   : {}".format(total_frames))
    print("Total working time       : {:.3f} s".format(working_time))
    print("Average per-frame latency: {:.3f} ms".format(per_frame_ms))
    print("Average overall FPS      : {:.2f} fps".format(avg_fps))
    if avg_fps >= 30.0:
        verdict = "SUITABLE for real-time streaming (>= 30 FPS)"
    elif avg_fps >= 24.0:
        verdict = "MARGINAL for streaming (24-30 FPS, near real-time)"
    else:
        verdict = "NOT SUITABLE for real-time streaming (< 24 FPS)"
    print("Streaming suitability    : {}".format(verdict))
    print("Note: measures detection head + SuppressNet on pre-extracted features.")
    print("      Real end-to-end streaming must also include backbone feature extraction.")
    print("==========================================================================\n")

    output_dict={"version":"VERSION 1.3","results":result_dict,"external_data":{}}
    outfile=open(opt["result_file"].format(opt['exp']),"w")
    json.dump(output_dict,outfile, indent=2)
    outfile.close()

    evaluation_detection(opt)


def main(opt):
    max_perf=0
    if opt['mode'] == 'train':
        max_perf=train(opt)
    if opt['mode'] == 'test':
        test(opt)
    if opt['mode'] == 'test_frame':
        test_frame(opt)
    if opt['mode'] == 'test_online':
        test_online(opt)
    if opt['mode'] == 'eval':
        evaluation_detection(opt)
        
    return max_perf

if __name__ == '__main__':
    opt = opts.parse_opt()
    opt = vars(opt)
    if not os.path.exists(opt["checkpoint_path"]):
        os.makedirs(opt["checkpoint_path"]) 
    opt_file=open(opt["checkpoint_path"]+"/opts.json","w")
    json.dump(opt,opt_file)
    opt_file.close()
    
    if opt['seed'] >= 0:
        seed = opt['seed'] 
        torch.manual_seed(seed)
        np.random.seed(seed)
        #random.seed(seed)
           
    opt['anchors'] = [int(item) for item in opt['anchors'].split(',')]  
           
    main(opt)
    while(opt['wterm']):
        pass
