#一些模块为了后面的可视化而增加
import os.path as osp
import os
from PIL import Image
import json
from matplotlib import pyplot as plt
#根据信息生成模板，此处要求bbox必须是xyxy标注
def coco_annotations(bbox,cid,bbox_id,img_id,iscrowd):
    x1,y1,x2,y2=bbox
    return {'segmentation':[[x1,y1,x2,y1,x2,y2,x1,y2]],
           'bbox':[x1,y1,x2-x1+1,y2-y1+1],
           'category_id':cid,
           'area':(y2-y1+1)*(x2-x1+1),
           'iscrowd':iscrowd,
           'image_id':img_id,
           'id':bbox_id}
def coco_images(file_name,height,width,img_id):
    return {'file_name':file_name,
           'height':height,
           'width':width,
           'id':img_id}

#从dota数据集中的txt格式标注中获取信息，函数中需要用到变量cls_name2id，categories
def deal_with_txt(label_path, img_id, anno_id):
    annos = []
    
    with open(label_path,'r') as gt:
        gt_lines=gt.readlines()
    for i in gt_lines:
        iscrowd=int(i[-2])
        i=i.split(' ')
        cls_name=i[-2]
        cid=cls_name2id[cls_name]
        
        cood=i[:8]
        cood=tuple(map(float,cood))
        x1=min(cood[0],cood[2],cood[4],cood[6])
        x2=max(cood[0],cood[2],cood[4],cood[6])
        y1=min(cood[1],cood[3],cood[5],cood[7])
        y2=max(cood[1],cood[3],cood[5],cood[7])
        b=(x1,y1,x2,y2)
        #下面使用coco_annotations函数把上面的读取数据输入
        anno=coco_annotations(b,cid,anno_id,img_id,iscrowd)
        annos.append(anno)
        anno_id+=1
    return annos,anno_id


def generate_coco_fmt(data_root,anno_root,categories,img_root):
    '''
    data_root是数据集路径
    下面的路径是相对data_root的路径
    anno_root是数据集的标注文件的路径
    img_root是数据集的图片路径
    
    categories需要事先得到
    '''
    img_id, anno_id = 0, 0
    all_annos, all_images = [], []
    
    for anno_txt in os.listdir(osp.join(data_root,anno_root)):
        file_name=anno_txt.replace('txt','png')
        label_path=osp.join(data_root,anno_root,anno_txt)
        img_path=osp.join(data_root,img_root,file_name)
        if osp.exists(img_path):
            annos,anno_id=deal_with_txt(label_path,img_id,anno_id)
            all_annos.extend(annos)
            
            img=Image.open(img_path)
            all_images.append(coco_images(osp.join(img_root,file_name),img.height,img.width,img_id))
            img_id+=1
    return{
        'images':all_images,
        "annotations": all_annos,
        "categories": categories,
        "type": "instance"
    }
'''
	这一部分的目的是为了获取必须提前获取的categories和cls_name2id
'''
classes=('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
               'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
               'basketball-court', 'storage-tank', 'soccer-ball-field',
               'roundabout', 'harbor', 'swimming-pool', 'helicopter')
cls_name2id={}
categories=[]
for i in range(1,16):
    cls_name2id.update({classes[i-1]:i})
    dict_cate={'id':i,'name':classes[i-1],'supercategory':classes[i-1]}
    categories.append(dict_cate)
#路径自己改一下记得
data_root='../autodl-pub/DOTA'
anno_root='trainval1024/annfiles'
img_root='trainval1024/images'
dota_coco_fmt=generate_coco_fmt(data_root,anno_root,categories,img_root)
json.dump(dota_coco_fmt,open(osp.join(data_root,'train.json'),'a'),ensure_ascii=False)
'''
    data_root是数据集路径
    下面的路径是相对data_root的路径
    anno_root是数据集的标注文件的路径
    img_root是数据集的图片路径
    
    categories需要事先得到
    '''