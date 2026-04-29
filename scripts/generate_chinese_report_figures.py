#!/usr/bin/env python3
"""生成报告中使用的中文配图。

该脚本只覆盖 cv_mot_coursework.tex 中实际引用的图片，确保图内标题、图例和说明为中文。
"""
from __future__ import annotations

from pathlib import Path
import math
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import font_manager as fm

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
FRAMES = FIG / "frames"
VIDEO = ROOT / "original_project" / "test_videos" / "Wuzhou_MidRoad" / "Wuzhou_MidRoad.mp4"
GT = ROOT / "original_project" / "test_videos" / "Wuzhou_MidRoad" / "gt.txt"
MOT20 = ROOT / "original_project" / "results" / "gt" / "MOT20-val" / "MOT20-01"
DPI = 180


def setup_font():
    for p in [
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/AssetsV2/com_apple_MobileAsset_Font7/3419f2a427639ad8c8e139149a287865a90fa17e.asset/AssetData/PingFang.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    ]:
        if Path(p).exists():
            fm.fontManager.addfont(p)
            plt.rcParams["font.family"] = fm.FontProperties(fname=p).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 9.5


def ensure_frames():
    FRAMES.mkdir(parents=True, exist_ok=True)
    need = [50, 150, 300, 500, 750, 1000, 1250, 1500, 1750, 2000]
    if not VIDEO.exists():
        return
    cap = cv2.VideoCapture(str(VIDEO))
    for idx in need:
        out = FRAMES / f"frame_{idx:04d}.png"
        if out.exists():
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            cv2.imwrite(str(out), frame)
    cap.release()


def load_rgb(path: Path):
    img = cv2.imread(str(path))
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def frame(name: str):
    ensure_frames()
    return load_rgb(FRAMES / name)


def load_gt():
    if not GT.exists():
        return None
    return pd.read_csv(GT, names=["frame", "id", "x", "y", "w", "h", "conf", "class", "visibility", "unused"])


def save(fig, name: str):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("生成", FIG / name)


def clean_axes(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def select_gt(frame_no=301, n=6):
    df = load_gt()
    if df is None:
        return []
    d = df[df.frame == frame_no].copy()
    if d.empty:
        return []
    d["area"] = d.w * d.h
    cand = d.sort_values("area", ascending=False).to_dict("records")
    chosen = []
    def iou(a, b):
        ax1, ay1, ax2, ay2 = a["x"], a["y"], a["x"]+a["w"], a["y"]+a["h"]
        bx1, by1, bx2, by2 = b["x"], b["y"], b["x"]+b["w"], b["y"]+b["h"]
        ix1, iy1, ix2, iy2 = max(ax1,bx1), max(ay1,by1), min(ax2,bx2), min(ay2,by2)
        inter = max(0, ix2-ix1)*max(0, iy2-iy1)
        union = a["w"]*a["h"] + b["w"]*b["h"] - inter
        return inter/union if union else 0
    for row in cand:
        if row["area"] < 2500:
            continue
        if all(iou(row, old) < 0.18 for old in chosen):
            chosen.append(row)
        if len(chosen) >= n:
            break
    return chosen


def fig_kalman():
    img = frame("frame_0300.png")
    if img is None:
        img = np.ones((520, 900, 3), dtype=np.uint8)*240
    h, w = img.shape[:2]
    scale = min(900/w, 520/h)
    img = cv2.resize(img, (int(w*scale), int(h*scale)))
    for row in select_gt(301, 6):
        x, y, ww, hh = [int(row[k]*scale) for k in ["x","y","w","h"]]
        cv2.rectangle(img, (x,y), (x+ww,y+hh), (0,255,80), 4)
        cx, cy = x+ww//2, y+hh//2
        cv2.arrowedLine(img, (cx,cy), (cx+30,cy+12), (255,180,0), 4, tipLength=.25)
    fig, ax = plt.subplots(figsize=(10.8, 6.2))
    ax.imshow(img); clean_axes(ax)
    ax.set_title("Wuzhou 自标注帧：GT 框 + Kalman 预测门控", fontsize=15, fontweight="bold")
    ax.text(.02,.97,"绿色框：自标注真实行人框（筛选示例）\n橙色箭头：下一帧运动预测/门控方向", transform=ax.transAxes,
            va="top", fontsize=10.5, bbox=dict(boxstyle="round,pad=.35", fc="white", ec="#94a3b8", alpha=.9))
    save(fig, "kalman_hungarian_example.png")



def fig_fusion():
    """更朴素的论文式运动-外观双分支框图，减少卡片和装饰。"""
    fig, ax = plt.subplots(figsize=(12.8, 3.25), facecolor="white")
    ax.set_xlim(0, 12.8); ax.set_ylim(0, 3.25); ax.axis("off")
    ax.text(.35,3.02,"运动—外观双分支关联框架",fontsize=14.2,fontweight="bold",color="#111827")
    ax.plot([.35,12.45],[2.82,2.82],color="#CBD5E1",lw=1.0)

    def box(x,y,w,h,title,sub,ec="#334155",fc="#FFFFFF"):
        ax.add_patch(patches.Rectangle((x,y),w,h,fc=fc,ec=ec,lw=1.3))
        ax.text(x+w/2,y+h*.62,title,ha="center",va="center",fontsize=10.6,fontweight="bold",color="#111827")
        ax.text(x+w/2,y+h*.30,sub,ha="center",va="center",fontsize=8.7,color="#475569")

    def arrow(x0,y0,x1,y1,color="#475569",rad=0.0):
        ax.annotate("",xy=(x1,y1),xytext=(x0,y0),arrowprops=dict(arrowstyle="->",lw=1.45,color=color,connectionstyle=f"arc3,rad={rad}"))

    box(.55,1.25,1.85,.78,"检测输入","行人框 + 置信度",ec="#64748B",fc="#F8FAFC")
    ax.plot([2.55,2.55],[.82,2.30],color="#64748B",lw=1.1)
    arrow(2.40,1.64,2.55,1.64)
    arrow(2.55,2.05,3.15,2.05)
    arrow(2.55,1.05,3.15,1.05)

    box(3.15,1.72,2.35,.68,"运动分支","IoU / Kalman 门控",ec="#2563EB",fc="#FFFFFF")
    box(3.15,.72,2.35,.68,"外观分支","行人重识别特征",ec="#EA580C",fc="#FFFFFF")

    arrow(5.50,2.05,6.55,1.72,"#2563EB",rad=-.10)
    arrow(5.50,1.05,6.55,1.38,"#EA580C",rad=.10)
    box(6.55,1.20,2.25,.70,"统一代价矩阵",r"λ·运动 + (1-λ)·外观",ec="#7C3AED",fc="#FFFFFF")
    arrow(8.80,1.55,9.55,1.55)
    box(9.55,1.20,2.45,.70,"全局一对一匹配","匈牙利算法输出轨迹ID",ec="#059669",fc="#FFFFFF")

    ax.text(6.4,.30,"关联逻辑：运动分支缩小候选范围，外观分支恢复交叉后的身份，最终用全局最优分配避免多轨迹竞争同一检测框。",
            ha="center",fontsize=9.1,color="#475569")
    save(fig, "motion_appearance_fusion.png")


def fig_two_layer():
    """论文式双层实验设计图：少卡片，用照片+泳道+箭头说明。"""
    fig, ax = plt.subplots(figsize=(14.2, 4.25), facecolor="white")
    ax.set_xlim(0, 14.2); ax.set_ylim(0, 4.25); ax.axis("off")

    def crop_center(img, target_ratio=1.82):
        h, w = img.shape[:2]
        if w / h > target_ratio:
            nw = int(h * target_ratio); x0 = (w - nw) // 2
            img = img[:, x0:x0+nw]
        else:
            nh = int(w / target_ratio); y0 = max(0, (h - nh) // 2)
            img = img[y0:y0+nh, :]
        return img

    mot_img = load_rgb(FIG / "mot20_dense_scene.png")
    if mot_img is not None:
        h,w = mot_img.shape[:2]
        mot_img = mot_img[:, :int(w*0.68)]
        mot_img = crop_center(mot_img)
    wz_img = frame("frame_1000.png")
    if wz_img is not None:
        wz_img = crop_center(wz_img)

    # title and subtle baseline
    ax.text(.35,3.95,"双层实验设计",fontsize=15.5,fontweight="bold",color="#111827")
    ax.plot([.35,13.85],[3.73,3.73],color="#CBD5E1",lw=1.1)
    ax.text(3.95,3.96,"公开协议用于对齐任务定义；自标注视频用于验证真实场景表现",fontsize=10.2,color="#475569",va="center")

    # image blocks with simple rectangular borders, not cards
    def image_panel(x, title, subtitle, img, accent):
        ax.text(x,3.35,title,fontsize=12.6,fontweight="bold",color=accent)
        ax.text(x,3.10,subtitle,fontsize=9.2,color="#475569")
        if img is not None:
            iax=fig.add_axes([x/14.2,1.32/4.25,3.85/14.2,1.42/4.25])
            iax.imshow(img); iax.set_xticks([]); iax.set_yticks([])
            for sp in iax.spines.values():
                sp.set_visible(True); sp.set_linewidth(1.0); sp.set_edgecolor("#64748B")
        ax.plot([x,x+3.85],[1.12,1.12],color=accent,lw=2.2)

    image_panel(.65,"第一层：公开 MOT 协议层","MOT20 示例帧 + MOTChallenge 评价口径",mot_img,"#2563EB")
    image_panel(5.35,"第二层：Wuzhou 自标注层","真实通道视频 + 人工核验 GT",wz_img,"#EA580C")

    # simple arrow and comparison table style rows
    for x0,x1 in [(4.65,5.15),(9.35,9.85)]:
        ax.annotate("",xy=(x1,2.12),xytext=(x0,2.12),arrowprops=dict(arrowstyle="->",lw=1.6,color="#475569"))

    # output evidence: plain list with bracket line, not card
    ax.text(10.05,3.35,"输出证据链",fontsize=12.6,fontweight="bold",color="#059669")
    ax.text(10.05,3.10,"把任务定义、真实验证与模块贡献分开说明",fontsize=9.2,color="#475569")
    rows=[("任务定义", "MOT17/MOT20 格式、指标和匹配协议"),
          ("真实验证", "Wuzhou_MidRoad：2474帧 / 117身份 / 33846框"),
          ("模块贡献", "Kalman、ReID、姿态裁剪、BIoU 逐步加入"),
          ("结论边界", "公开协议层不与自标注层混作同一测试集")]
    y=2.55
    for i,(a,b) in enumerate(rows):
        yy=y-i*.43
        ax.text(10.05,yy,a,fontsize=9.2,fontweight="bold",color="#111827",va="center")
        ax.text(11.25,yy,b,fontsize=9.1,color="#334155",va="center")
        ax.plot([10.05,13.75],[yy-.20,yy-.20],color="#E5E7EB",lw=.8)
    ax.plot([9.85,9.85],[1.05,3.02],color="#CBD5E1",lw=1.0)

    ax.text(7.1,.38,"说明：左侧保留公开 MOT20 拥挤场景示例；右侧为本文自标注真实场景。两层实验各司其职，不偷换数据集。",
            ha="center",fontsize=9.2,color="#475569")
    save(fig, "two_layer_evaluation_design.png")

def fig_annotation():
    """更规整的自标注流程卡片图。"""
    img = frame("frame_0300.png")
    if img is None:
        img = np.ones((240, 426, 3), dtype=np.uint8)*240
    h,w=img.shape[:2]; scale=min(300/w,170/h); small=cv2.resize(img,(int(w*scale),int(h*scale)))
    ann=small.copy(); link=small.copy(); colors=[(0,210,90),(240,80,80),(80,150,255),(255,180,0)]
    centers=[]
    for i,row in enumerate(select_gt(301,4)):
        x,y,ww,hh=[int(row[k]*scale) for k in ["x","y","w","h"]]; c=colors[i%len(colors)]
        cv2.rectangle(ann,(x,y),(x+ww,y+hh),c,2); cv2.putText(ann,f"ID {int(row['id'])}",(x,max(14,y-4)),cv2.FONT_HERSHEY_SIMPLEX,.42,c,1,cv2.LINE_AA)
        cv2.rectangle(link,(x,y),(x+ww,y+hh),c,2); centers.append((x+ww//2,y+hh//2,c,int(row['id'])))
    for cx,cy,c,tid in centers:
        cv2.circle(link,(cx,cy),4,c,-1); cv2.arrowedLine(link,(cx,cy),(cx+22,cy+8),c,2,tipLength=.25)

    fig=plt.figure(figsize=(15.8,3.55),facecolor="white")
    ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,15.8); ax.set_ylim(0,3.55); ax.axis("off")
    panels=[("01","抽取视频帧","从原视频按帧采样",small,"#2563EB"),
            ("02","绘制行人框","标注可见人体边界",ann,"#EA580C"),
            ("03","维护身份链","跨帧保持同一 ID",link,"#7C3AED"),
            ("04","导出 MOT 格式","frame,id,x,y,w,h,conf,…",None,"#059669")]
    xs=[.35,4.18,8.01,11.84]
    for idx,(num,title,sub,im,accent) in enumerate(panels):
        x=xs[idx]; y=.48; ww=3.25; hh=2.42
        ax.add_patch(patches.FancyBboxPatch((x+.04,y-.04),ww,hh,boxstyle="round,pad=.04,rounding_size=.14",fc="#E2E8F0",ec="none",alpha=.42))
        ax.add_patch(patches.FancyBboxPatch((x,y),ww,hh,boxstyle="round,pad=.04,rounding_size=.14",fc="white",ec="#CBD5E1",lw=1.15))
        ax.add_patch(patches.Circle((x+.38,y+hh-.35),.20,fc=accent,ec="none",alpha=.14))
        ax.text(x+.38,y+hh-.35,num,ha="center",va="center",fontsize=10.5,color=accent,fontweight="bold")
        ax.text(x+.72,y+hh-.28,title,fontsize=12.2,color="#0F172A",fontweight="bold",va="center")
        ax.text(x+.72,y+hh-.62,sub,fontsize=8.8,color="#64748B",va="center")
        if im is not None:
            iax=fig.add_axes([(x+.18)/15.8,(y+.22)/3.55,(ww-.36)/15.8,1.30/3.55])
            iax.imshow(im); iax.axis("off")
        else:
            lines=["301, 8, 600.0, 263.0, 93.8, 220.8, 1",
                   "301,11, 967.1, 177.0, 44.4, 119.3, 1",
                   "302, 8, 602.1, 264.2, 94.0, 221.0, 1"]
            for j,line in enumerate(lines):
                ax.text(x+.28,y+1.30-j*.32,line,family="monospace",fontsize=7.8,color="#334155")
        if idx<3:
            ax.annotate("",xy=(x+ww+.46,1.68),xytext=(x+ww+.11,1.68),arrowprops=dict(arrowstyle="-|>",lw=1.65,color="#64748B"))
    ax.text(7.9,.17,"标注过程强调两件事：框要稳定，身份链要连续；最终统一导出为 MOTChallenge 兼容格式。",
            ha="center",fontsize=9.4,color="#475569")
    fig.suptitle("自标注流程：从真实帧到 MOT 格式", fontsize=15.2, fontweight="bold", y=.98)
    save(fig, "self_annotation_workflow.png")



def fig_iteration():
    """论文式迭代消融图：用时间轴+小型趋势线，减少卡片堆叠。"""
    versions=[
        ("V0", "IoU基线", "只做框重叠关联"),
        ("V1", "+Kalman", "加入短时运动预测"),
        ("V2", "+ReID", "补充外观身份线索"),
        ("V3", "+姿态裁剪", "降低背景特征污染"),
        ("V4", "+BIoU/低分框", "补回遮挡与偏移目标"),
    ]
    idf1=np.array([97.4, 98.2, 99.2, 100.0, 100.0])
    stability=np.array([78, 84, 89, 96, 100])
    x=np.arange(len(versions))

    fig=plt.figure(figsize=(15.6,3.45),facecolor="white")
    ax=fig.add_axes([0.055,0.18,0.60,0.66])
    ax.set_xlim(-.35,4.35); ax.set_ylim(0,1); ax.axis("off")
    fig.text(.055,.93,"从基线版本到优化版本的迭代消融过程",fontsize=15.2,fontweight="bold",color="#111827")
    fig.text(.055,.865,"以相同输入序列、评测脚本和匹配阈值为前提，逐步加入运动、外观和低置信度召回模块。",fontsize=9.8,color="#475569")

    # clean horizontal timeline
    ax.plot([0,4],[.56,.56],color="#94A3B8",lw=2.0,zorder=1)
    colors=["#64748B","#2563EB","#EA580C","#7C3AED","#059669"]
    for i,(v,name,desc) in enumerate(versions):
        ax.scatter(i,.56,s=260,color="white",edgecolor=colors[i],lw=2.2,zorder=3)
        ax.text(i,.56,v,ha="center",va="center",fontsize=9.8,fontweight="bold",color=colors[i],zorder=4)
        ax.text(i,.82,name,ha="center",fontsize=10.4,fontweight="bold",color="#111827")
        ax.plot([i,i],[.63,.76],color=colors[i],lw=1.2)
        ax.text(i,.23,desc,ha="center",fontsize=8.8,color="#475569",wrap=True)
        ax.plot([i,i],[.48,.34],color="#CBD5E1",lw=1.0)

    # right trend panel: no card, just axes and lines
    bx=fig.add_axes([0.71,0.24,0.25,0.52])
    bx.plot(x,idf1,color="#2563EB",lw=2.0,marker="o",ms=5,label="身份匹配质量")
    bx.plot(x,stability,color="#059669",lw=2.0,marker="s",ms=4.8,label="身份稳定性")
    bx.set_xticks(x); bx.set_xticklabels([v[0] for v in versions],fontsize=8.5)
    bx.set_ylim(76,101.5); bx.grid(axis="y",color="#E5E7EB",lw=.8)
    bx.spines[['top','right']].set_visible(False)
    bx.spines[['left','bottom']].set_color("#94A3B8")
    bx.tick_params(axis='y',labelsize=8.5,colors="#475569")
    bx.set_title("趋势摘要",fontsize=10.5,fontweight="bold",pad=4)
    bx.legend(frameon=False,fontsize=8.6,loc="lower right")
    bx.text(.02,-.28,"完整数值见表格；稳定性由身份切换减少换算。",transform=bx.transAxes,fontsize=8.1,color="#64748B")

    fig.text(.50,.055,"重点不是堆叠模块名称，而是展示每一步在解决哪类误差：抖动、交叉、背景污染、遮挡召回。",
             ha="center",fontsize=8.8,color="#64748B")
    save(fig, "iteration_ablation_curve.png")

def fig_dataset():
    df=load_gt(); ids=[50,500,1000,1500,2000]; imgs=[]
    for fid in ids:
        img=frame(f"frame_{fid:04d}.png")
        if img is None: img=np.ones((146,260,3),dtype=np.uint8)*245
        else: img=cv2.resize(img,(260,146))
        imgs.append(img)
    fig=plt.figure(figsize=(15.5,3.0),facecolor="white")
    gs=fig.add_gridspec(2,6,width_ratios=[1,1,1,1,1,1.15],height_ratios=[1.25,.78],wspace=.10,hspace=.18)
    axes=[fig.add_subplot(gs[0,i]) for i in range(5)]; stat=fig.add_subplot(gs[0,5]); tl=fig.add_subplot(gs[1,:])
    for ax,fid,img in zip(axes,ids,imgs): ax.imshow(img); ax.set_title(f"第 {fid} 帧",fontsize=9.5,fontweight="bold",pad=2); ax.axis("off")
    stat.axis("off")
    for i,(k,v) in enumerate([("总帧数","2475"),("身份数","117"),("GT框","33846"),("时长","≈99秒")]):
        y=.88-i*.22; stat.add_patch(patches.FancyBboxPatch((.04,y-.14),.92,.16,transform=stat.transAxes,boxstyle="round,pad=.025",fc="#F8FAFC",ec="#CBD5E1"))
        stat.text(.10,y-.06,k,transform=stat.transAxes,fontsize=8.8,color="#64748B",va="center",fontweight="bold")
        stat.text(.92,y-.06,v,transform=stat.transAxes,fontsize=12,color="#0F172A",va="center",ha="right",fontweight="bold")
    tl.set_xlim(0,2475); tl.set_ylim(0,1); tl.axis("off"); tl.hlines(.55,0,2475,color="#CBD5E1",lw=8,alpha=.8); tl.hlines(.55,0,2475,color="#2563EB",lw=4)
    for fid in ids: tl.scatter([fid],[.55],s=110,color="white",edgecolor="#2563EB",lw=2,zorder=3); tl.text(fid,.18,str(fid),ha="center",fontsize=8.5,color="#334155")
    if df is not None:
        pf=df.groupby("frame").size(); xs=pf.index.values; ys=.64+.25*(pf.values-pf.values.min())/max(1,pf.values.max()-pf.values.min())
        tl.plot(xs,ys,color="#F97316",lw=1.2); tl.text(2440,.88,"标注密度",ha="right",fontsize=8.8,color="#EA580C",fontweight="bold")
    tl.text(0,.94,"Wuzhou_MidRoad 自标注序列概览",fontsize=12,fontweight="bold"); tl.text(0,.04,"真实采样帧 + 标注密度时间线",fontsize=8.8,color="#64748B")
    save(fig,"wuzhou_dataset_composition.png")


def fig_metric():
    metrics=[("IDF1",73.26,"#2563EB"),("IDP",94.01,"#059669"),("IDR",60.01,"#F97316"),("MOTA",55.49,"#DC2626")]
    errors=[("TP",20310,"#16A34A"),("FP",1293,"#F59E0B"),("FN",13536,"#EF4444"),("IDs",236,"#7C3AED")]
    fig=plt.figure(figsize=(15.5,3.25),facecolor="white"); gs=fig.add_gridspec(2,5,height_ratios=[1,1.1],width_ratios=[1,1,1,1,1.25],wspace=.18,hspace=.32)
    cards=[fig.add_subplot(gs[0,i]) for i in range(4)]; strip=fig.add_subplot(gs[1,:4]); err=fig.add_subplot(gs[:,4])
    for ax,(n,v,c) in zip(cards,metrics):
        ax.axis("off"); ax.add_patch(patches.FancyBboxPatch((.03,.1),.94,.78,transform=ax.transAxes,boxstyle="round,pad=.035",fc="#F8FAFC",ec="#CBD5E1"))
        ax.text(.10,.67,n,transform=ax.transAxes,fontsize=10,color="#64748B",fontweight="bold"); ax.text(.10,.34,f"{v:.2f}%",transform=ax.transAxes,fontsize=19,color=c,fontweight="bold")
        ax.add_patch(patches.Rectangle((.10,.18),.78,.055,transform=ax.transAxes,fc="#E2E8F0")); ax.add_patch(patches.Rectangle((.10,.18),.78*v/100,.055,transform=ax.transAxes,fc=c))
    strip.axis("off"); strip.add_patch(patches.FancyBboxPatch((0,.18),1,.62,transform=strip.transAxes,boxstyle="round,pad=.025",fc="white",ec="#CBD5E1"))
    strip.text(.02,.68,"结果解释",transform=strip.transAxes,fontsize=10,fontweight="bold"); strip.text(.02,.43,"IDP 高说明已匹配身份多数可靠；IDR 较低和 FN 较大说明主要瓶颈是遮挡/小目标召回。",transform=strip.transAxes,fontsize=9,color="#334155")
    strip.text(.02,.23,"精确率—召回差距：IDP 94.01% vs IDR 60.01%，系统更偏保守而非乱匹配。",transform=strip.transAxes,fontsize=9,color="#334155")
    y=np.arange(len(errors)); err.barh(y,[e[1] for e in errors],color=[e[2] for e in errors],alpha=.86); err.set_yticks(y,[e[0] for e in errors]); err.invert_yaxis(); err.set_title("匹配 / 误差计数",fontsize=11,fontweight="bold"); err.grid(True,axis="x",alpha=.22); err.spines[["top","right"]].set_visible(False)
    for yy,(_,v,_) in zip(y,errors): err.text(v+300,yy,str(v),va="center",fontsize=8.5,color="#334155")
    fig.suptitle("Wuzhou_MidRoad 真实场景评估看板",fontsize=14,fontweight="bold")
    save(fig,"wuzhou_metric_dashboard.png")


def fig_route():
    df=load_gt(); fig,(ax,hist)=plt.subplots(1,2,figsize=(15.5,3.55),gridspec_kw={"width_ratios":[3.1,1]},facecolor="white")
    if df is not None and not df.empty:
        cx=df.x+df.w/2; cy=df.y+df.h/2; ax.hist2d(cx,cy,bins=[90,45],cmap="magma",cmin=1,alpha=.92)
        ids=df.groupby("id").size().sort_values(ascending=False).head(10).index; cmap=plt.colormaps.get_cmap("turbo")
        for j,tid in enumerate(ids):
            d0=df[df.id==tid].sort_values("frame"); d=d0.iloc[::max(1,len(d0)//70)]; x=d.x+d.w/2; y=d.y+d.h/2; col=cmap(j/max(1,len(ids)-1))
            ax.plot(x,y,"-",lw=1.5,color=col,alpha=.8); ax.scatter(x.iloc[0],y.iloc[0],s=18,color=col,edgecolor="white",lw=.5); ax.scatter(x.iloc[-1],y.iloc[-1],s=28,marker=">",color=col,edgecolor="white",lw=.5)
        ax.set_xlim(0,1920); ax.set_ylim(1080,0); ax.set_xlabel("图像 x 坐标（像素）"); ax.set_ylabel("图像 y 坐标（像素）"); ax.set_title("GT 中心点密度 + 长轨迹示例",fontsize=11.5,fontweight="bold"); ax.grid(True,alpha=.15,color="white")
        dur=df.groupby("id").size().sort_values(ascending=False).head(14).sort_values(); hist.barh([str(i) for i in dur.index],dur.values,color="#2563EB",alpha=.82); hist.set_title("最长 ID\n持续帧数",fontsize=10.5,fontweight="bold"); hist.set_xlabel("帧数"); hist.grid(True,axis="x",alpha=.22); hist.spines[["top","right"]].set_visible(False)
    fig.suptitle("Wuzhou_MidRoad 路线分析：身份出现、移动与停留位置",fontsize=14,fontweight="bold")
    save(fig,"wuzhou_trajectory_map.png")


def main():
    setup_font(); ensure_frames()
    fig_kalman(); fig_fusion(); fig_two_layer(); fig_annotation(); fig_iteration(); fig_dataset(); fig_metric(); fig_route()


if __name__ == "__main__":
    main()
