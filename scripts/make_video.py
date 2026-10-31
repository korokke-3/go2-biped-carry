"""Build the final-assignment video (1920x1080, 30 fps) from the real-robot recording, simulation clips and title /
explanation slides drawn with PIL, with narration from a local VOICEVOX engine (http://127.0.0.1:50021); ffmpeg from
imageio-ffmpeg.

  python scripts/make_video.py --real <phone video.mp4> --mujoco <sim2sim clip.mp4> --out outputs/final_video.mp4

Each part lasts as long as its narration needs: 2x clips slow down towards 1x first, then hold their last frame.

The phone video is stored portrait with a rotation tag; ffmpeg auto-rotates it and `transpose=2` turns it upright.
Real-video seconds 16-24 show a person's face and are not used.
"""
import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import urllib.parse
import urllib.request
import wave

import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FF = imageio_ffmpeg.get_ffmpeg_exe()
W, H, FPS = 1920, 1080, 30
FONT_R = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_B = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
BG, FG, ACC, SUB = (14, 22, 36), (240, 243, 247), (255, 184, 64), (150, 170, 195)

HB = "logs/rsl_rl/unitree_go2_biped_holdball_robust_fine/2026-10-25_08-49-12_holdball_c10/eval_videos/"
SIM = {
    "gentle": HB + "model_23984_biped_final_row1_gentle-step-0.mp4",
    "drop": HB + "model_23984_biped_final_row0_drop10-step-0.mp4",
    "carry": HB + "model_23984_biped_final_group4_s36_carry-step-0.mp4",
}


def font(size, bold=False):
    return ImageFont.truetype(FONT_B if bold else FONT_R, size)


def text_w(d, s, f):
    return d.textlength(s, font=f)


def wrap(d, s, f, width):
    """Greedy line breaks by pixel width (character level, Japanese has no spaces)."""
    lines, cur = [], ""
    for ch in s:
        if text_w(d, cur + ch, f) > width and cur and ch not in "、。,.)」』・:":  # no punctuation at a line start
            lines.append(cur)
            cur = ch
        else:
            cur += ch
    return lines + [cur]


def overlay(caption=None, tag=None, speed=None, big=None, sub=None):
    """Transparent 1920x1080 overlay: lower-third caption (list of lines), top-left tag, top-right speed badge,
    centred big text."""
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    if tag:
        f = font(34, True)
        tw = text_w(d, tag, f)
        d.rounded_rectangle([40, 36, 40 + tw + 44, 36 + 62], 12, fill=(14, 22, 36, 215))
        d.text((62, 42), tag, font=f, fill=ACC)
    if speed:
        f = font(44, True)
        s = f"×{speed:g} 倍速"
        tw = text_w(d, s, f)
        d.rounded_rectangle([W - tw - 90, 36, W - 40, 36 + 70], 12, fill=(200, 40, 40, 230))
        d.text((W - tw - 66, 40), s, font=f, fill=(255, 255, 255))
    if caption:
        f = font(50, True)
        lines = caption if isinstance(caption, list) else [caption]
        lh = 70
        top = H - 60 - lh * len(lines) - 30
        d.rectangle([0, top, W, H], fill=(0, 0, 0, 165))
        for i, ln in enumerate(lines):
            fi = f if i == 0 else font(40)
            tw = text_w(d, ln, fi)
            d.text(((W - tw) / 2, top + 26 + i * lh), ln, font=fi, fill=FG if i == 0 else (220, 225, 232))
    if big:
        f = font(84, True)
        tw = text_w(d, big, f)
        y0 = 130  # upper part: the robot and the hand-over stay visible
        d.rectangle([0, y0, W, y0 + (230 if sub else 150)], fill=(0, 0, 0, 170))
        d.text(((W - tw) / 2, y0 + 20), big, font=f, fill=FG)
        if sub:
            fs = font(46)
            tw = text_w(d, sub, fs)
            d.text(((W - tw) / 2, y0 + 145), sub, font=fs, fill=ACC)
    return im


def slide(title, items, note=None, kicker=None):
    """Full slide: title, bullet items (str or (str, [sub-bullets])), optional footnote."""
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 18, H], fill=ACC)
    y = 70
    if kicker:
        d.text((110, y), kicker, font=font(36, True), fill=ACC)
        y += 60
    d.text((110, y), title, font=font(68, True), fill=FG)
    y += 130
    for it in items:
        main, subs = (it, []) if isinstance(it, str) else it
        d.ellipse([118, y + 24, 136, y + 42], fill=ACC)
        for ln in wrap(d, main, font(46, True), W - 165 - 100):
            d.text((165, y), ln, font=font(46, True), fill=FG)
            y += 66
        y += 6
        for s in subs:
            for k, ln in enumerate(wrap(d, s, font(36), W - 230 - 100)):
                d.text((200 if k == 0 else 230, y), ("– " if k == 0 else "") + ln, font=font(36), fill=(205, 214, 226))
                y += 52
        y += 18
    if note:
        d.text((110, H - 80), note, font=font(30), fill=SUB)
    return im


PANEL_W = 840                      # split layout: text panel on the left, footage on the right
VID_W, VID_H = 1072, 602  # even sizes for yuv420p
VID_X, VID_Y = PANEL_W + (W - PANEL_W - VID_W) // 2, (H - VID_H) // 2


def split_overlay(title, items, kicker=None, tag=None):
    """Left text panel (opaque) + transparent window for the footage on the right."""
    im = Image.new("RGBA", (W, H), BG + (255,))
    d = ImageDraw.Draw(im)
    d.rectangle([VID_X, VID_Y, VID_X + VID_W - 1, VID_Y + VID_H - 1], fill=(0, 0, 0, 0))
    d.rectangle([0, 0, 14, H], fill=ACC)
    x, tw, y = 70, PANEL_W - 70 - 40, 90
    if kicker:
        d.text((x, y), kicker, font=font(32, True), fill=ACC)
        y += 54
    for ln in wrap(d, title, font(54, True), tw):
        d.text((x, y), ln, font=font(54, True), fill=FG)
        y += 74
    y += 36
    for it in items:
        main, subs = (it, []) if isinstance(it, str) else it
        d.ellipse([x + 4, y + 20, x + 18, y + 34], fill=ACC)
        for ln in wrap(d, main, font(36, True), tw - 36):
            d.text((x + 36, y), ln, font=font(36, True), fill=FG)
            y += 52
        for sb in subs:
            for ln in wrap(d, sb, font(29), tw - 60):
                d.text((x + 60, y), ln, font=font(29), fill=(205, 214, 226))
                y += 42
        y += 22
    if tag:
        f = font(28, True)
        w = text_w(d, tag, f)
        d.rounded_rectangle([VID_X, VID_Y - 58, VID_X + w + 36, VID_Y - 10], 10, fill=(30, 42, 62, 255))
        d.text((VID_X + 18, VID_Y - 55), tag, font=f, fill=ACC)
    return im


def title_slide():
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 18, H], fill=ACC)
    d.text((140, 250), "PAI 最終課題", font=font(44, True), fill=ACC)
    d.text((140, 340), "後ろ足で立ち、", font=font(96, True), fill=FG)
    d.text((140, 470), "前腕で荷物を受け止めて運ぶ四足ロボット", font=font(80, True), fill=FG)
    d.text((140, 640), "Unitree Go2 × 強化学習 × 実機", font=font(48), fill=SUB)
    return im


def run(cmd):
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def fit_filter(rotate=False, speed=1.0):
    f = ("transpose=2," if rotate else "") + f"setpts=PTS/{speed},"
    return f + (f"scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=0x0e1624,"
                f"setsar=1,fps={FPS}")


def render_clip(src, start, dur, out, ov, rotate=False, speed=1.0, hold=0.0, tmp="."):
    """dur: seconds of source; output length dur / speed + hold (last frame held)."""
    png = os.path.join(tmp, os.path.basename(out) + ".png")
    ov.save(png)
    L = dur / speed + hold
    fade = f"fade=t=in:st=0:d=0.3,fade=t=out:st={L - 0.3:.3f}:d=0.3"
    pad = f",tpad=stop_mode=clone:stop_duration={hold:.3f}" if hold > 0 else ""
    run([FF, "-y", "-ss", str(start), "-t", str(dur), "-i", src, "-i", png, "-filter_complex",
         f"[0:v]{fit_filter(rotate, speed)}{pad}[v];[v][1:v]overlay=0:0,{fade},format=yuv420p[o]", "-map", "[o]",
         "-an", "-t", f"{L:.3f}", "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-r", str(FPS), out])
    return L


def render_split(segs, need, out, ov, tmp=".", min_dur=6.0):
    """Footage segments (src, st, dur, rotate) played one after another in the right window for as long as the
    narration needs (at least min_dur s, reading time), cut there or the last frame held; ov: split_overlay."""
    png = os.path.join(tmp, os.path.basename(out) + ".png")
    ov.save(png)
    total = sum(sg["dur"] for sg in segs)
    L = max(need, min_dur)
    cmd, chains = [FF, "-y"], []
    for k, sg in enumerate(segs):
        src = sg["src"] if os.path.isabs(sg["src"]) else os.path.join(ROOT, sg["src"])
        cmd += ["-ss", str(sg["st"]), "-t", str(sg["dur"]), "-i", src]
        rot = "transpose=2," if sg.get("rotate") else ""
        chains.append(f"[{k}:v]{rot}setpts=PTS-STARTPTS,scale={VID_W}:{VID_H}:force_original_aspect_ratio=decrease,"
                      f"pad={VID_W}:{VID_H}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={FPS}[s{k}]")
    n = len(segs)
    cmd += ["-i", png]
    fc = ";".join(chains) + ";" + "".join(f"[s{k}]" for k in range(n)) + f"concat=n={n}:v=1:a=0[c];"
    fc += (f"[c]tpad=stop_mode=clone:stop_duration={max(0.0, L - total) + 0.1:.3f},"
           f"pad={W}:{H}:{VID_X}:{VID_Y}:color=0x0e1624[b];[b][{n}:v]overlay=0:0,"
           f"fade=t=in:st=0:d=0.3,fade=t=out:st={L - 0.3:.3f}:d=0.3,format=yuv420p[o]")
    run(cmd + ["-filter_complex", fc, "-map", "[o]", "-an", "-t", f"{L:.3f}", "-c:v", "libx264", "-preset", "medium",
               "-crf", "18", "-r", str(FPS), out])
    return L


def render_still(im, dur, out, tmp="."):
    png = os.path.join(tmp, os.path.basename(out) + ".png")
    im.convert("RGB").save(png)
    run([FF, "-y", "-loop", "1", "-t", str(dur), "-i", png, "-vf",
         f"fps={FPS},fade=t=in:st=0:d=0.3,fade=t=out:st={dur - 0.3:.3f}:d=0.3,format=yuv420p",
         "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-tune", "stillimage", out])
    return dur


VV = "http://127.0.0.1:50021"
SPEAKER = 8           # VOICEVOX: 春日部つむぎ (ノーマル)
SPEAKER_NAME = "春日部つむぎ"
SPEED = 1.2
LEAD, TAIL = 0.35, 0.45  # narration starts LEAD s into a part; at least TAIL s of silence after it


def tts(text, path):
    """VOICEVOX audio_query + synthesis -> wav (cached by text)."""
    if os.path.exists(path):
        return
    q = urllib.request.urlopen(urllib.request.Request(
        f"{VV}/audio_query?" + urllib.parse.urlencode({"text": text, "speaker": SPEAKER}), method="POST")).read()
    query = json.loads(q)
    query.update({"speedScale": SPEED, "prePhonemeLength": 0.05, "postPhonemeLength": 0.1, "volumeScale": 1.2})
    wav = urllib.request.urlopen(urllib.request.Request(
        f"{VV}/synthesis?speaker={SPEAKER}", data=json.dumps(query).encode(),
        headers={"Content-Type": "application/json"}, method="POST")).read()
    with open(path, "wb") as f:
        f.write(wav)


def wav_len(path):
    with wave.open(path) as w:
        return w.getnframes() / w.getframerate()


C16 = "logs/rsl_rl/unitree_go2_biped_holdv4_fine/2026-10-24_02-35-01_holdv4_c16/eval_videos/"


def build_parts(R, MJ):
    REAL, ISAAC, MUJ = "実機 Unitree Go2", "シミュレーション Isaac Lab", "シミュレーション MuJoCo"
    return [
        dict(kind="clip", src=R, st=27.5, dur=6.0, rotate=True,
             ov=dict(tag=REAL, big="重さも形も分からない荷物を、前腕で受け止める", sub="水 2 L 入りのカバン。学習していない形"),
             narr="重さも形も分からない荷物を、前足で受け止める。"),
        dict(kind="still", im=title_slide(), dur=4.0,
             narr="四足ロボット、ゴーツーを後ろ足で立たせ、荷物を運ばせました。"),
        dict(kind="split", segs=[dict(src=R, st=25.0, dur=15.0, rotate=True)],
             ov=dict(kicker="課題設定", title="やりたいこと", tag=REAL, items=[
                 "後ろ足で立ち、前足を腕として使う",
                 ("重さも形も分からない荷物を載せられても、倒れない", ["そっと置かれても、急に置かれても"]),
                 "持ったまま、指示した方向へ運ぶ",
                 ("荷物がないときは四足で休む", ["立ち続けると、後ろ足のモーターが熱くなる"]),
             ]),
             narr="目標は、後ろ足で立って前足を腕として使い、重さの分からない荷物を載せられても倒れずに、"
                  "指示した方向へ運ぶこと。荷物がないときは四足に下りて、モーターを休ませます。"),
        dict(kind="split", segs=[dict(src=C16 + "model_31479_biped_group0_carry-step-0.mp4", st=0.0, dur=10.0),
                                 dict(src=C16 + "model_31479_biped_row_drop10-step-0.mp4", st=0.0, dur=6.0)],
             ov=dict(kicker="方法", title="強化学習で、1つの方策を学習する", tag=ISAAC, items=[
                 ("Isaac Lab で 4096 台を同時に学習", ["アルゴリズムは PPO"]),
                 ("方策が見るのは関節と IMU だけ", ["荷物の重さは教えない"]),
                 ("二足歩行は公開実装 go2_quad2hand を参考に", ["前足で立つ逆立ち歩行を、後ろ足立ちに読み替えた"]),
                 ("少しずつ難しくする", ["荷物 0.3 → 2 kg、直径 15〜46 cm の箱と球"]),
                 ("現実とのずれに強くする", ["摩擦・重さ・モーターの強さ・実機で測った関節の摩擦をランダムに変える"]),
             ]),
             narr="アイザックラボで四千台あまりを同時に動かし、強化学習で一つの方策を学習しました。"
                  "方策は関節と体の傾きだけを見て、荷物の重さは知りません。"
                  "荷物の重さと大きさは少しずつ難しくし、摩擦やモーターの強さもランダムに変えました。"),
        dict(kind="clip", src=SIM["gentle"], st=0.0, dur=6.0,
             ov=dict(tag=ISAAC, caption=["箱と球を前腕にそっと置く。重さは 2 kg", "置き方・大きさ・硬さはランダム"]),
             narr="大きさや硬さの違う箱と球を、前腕に置いていきます。置き方も毎回変えています。"),
        dict(kind="clip", src=SIM["drop"], st=0.0, dur=6.0,
             ov=dict(tag=ISAAC, caption=["10 cm の高さから落としても受け止める"]),
             narr="十センチの高さから落としても、受け止めます。"),
        dict(kind="clip", src=SIM["carry"], st=1.0, dur=8.0,
             ov=dict(tag=ISAAC, caption=["荷物を持ったまま、指示した方向へ歩く"]),
             narr="持ったまま、指示した方向へ歩きます。腕の間より小さい箱は、まだ落としてしまいます。"),
        dict(kind="clip", src=MJ, st=0.0, dur=14.0, speed=2.0,
             ov=dict(tag=MUJ, caption=["別の物理エンジンでも確かめる", "実機で測った関節の摩擦を入れて、立つ → 歩く → 旋回 → 四足に下りる"]),
             narr="別の物理エンジンでも、実機で測った関節の摩擦を入れて確かめました。"),
        dict(kind="clip", src=R, st=0.0, dur=4.0, rotate=True,
             ov=dict(tag=REAL, caption=["四足から、後ろ足で立ち上がる"]),
             narr="ここから実機です。後ろ足で立ち上がります。"),
        dict(kind="clip", src=R, st=7.5, dur=8.0, rotate=True, speed=2.0,
             ov=dict(tag=REAL, caption=["止まっていても足踏みでバランスを取る", "リモコンで前後・横・旋回"]),
             narr="足踏みでバランスを取りながら、リモコンで動かせます。"),
        dict(kind="clip", src=R, st=25.0, dur=10.0, rotate=True,
             ov=dict(tag=REAL, caption=["人が荷物を前腕に載せる", "学習したのは箱と球だけ。カバンは初めての形"]),
             narr="水二リットル入りのカバンを載せます。学習していない形ですが、しっかり受け止めました。"),
        dict(kind="clip", src=R, st=35.0, dur=13.0, rotate=True, speed=2.0,
             ov=dict(tag=REAL, caption=["持ったまま旋回・横へ移動"]),
             narr="持ったまま向きを変え、横にも動きます。荷物の重さは、体の揺れから読み取っています。"),
        dict(kind="clip", src=R, st=48.0, dur=16.0, rotate=True, speed=2.0,
             ov=dict(tag=REAL, caption=["持ったまま前後・旋回"]),
             narr="前後にも運べます。止まれの指示のときも、足踏みでバランスを保ちます。"),
        dict(kind="clip", src=R, st=64.0, dur=15.0, rotate=True, speed=2.0,
             ov=dict(tag=REAL, caption=["カメラの前まで運んで、戻る"]),
             narr="カメラの前まで運んで、戻ります。"),
        dict(kind="clip", src=R, st=79.0, dur=8.0, rotate=True,
             ov=dict(tag=REAL, caption=["荷物を受け取ったら、四足に下りて休む", "後ろ足のモーターの発熱を抑える"]),
             narr="荷物を取ると四足に下りて、熱くなった後ろ足のモーターを休ませます。"),
        dict(kind="split", segs=[dict(src=MJ, st=0.0, dur=14.0)],
             ov=dict(kicker="sim-to-real", title="実機に載せるまでの工夫", tag=MUJ, items=[
                 ("2つのシミュレーターで確かめる", ["MuJoCo で倒れた原因を、学習に戻した"]),
                 "実機と同じ通信で、制御プログラムを通しで試す",
                 ("四足に下りる動きは、0.5 秒の決めた動きで補う", ["MuJoCo で 36 回中 36 回成功"]),
                 ("勝手に前へ進んだ → 傾きの観測を補正", ["傾き 3° のずれで 0.3 m/s 流れることを確認"]),
                 ("安全装置", ["傾き・関節の範囲・通信の途絶で脱力。70 °C で自動で四足へ"]),
             ]),
             narr="実機に載せる前に、二つのシミュレーターと、実機と同じ通信の試験で確かめました。"
                  "実機で勝手に前へ進んだときは、記録から傾きのずれを見つけて補正しました。"),
        dict(kind="split", segs=[dict(src=R, st=48.0, dur=16.0, rotate=True)], min_dur=10.0,
             ov=dict(kicker="結果", title="結果", tag=REAL, items=[
                 ("シミュレーション、荷物 2 kg", ["転倒 0%", "球 30・36・44 cm を落とす割合:0・0・6%", "前腕にまたがる箱:0〜3%"]),
                 ("実機", ["立ち上がり 11 回、二足で約 11 分", "学習していない形のカバンを持って、旋回・前後・横に移動",
                          "モーターは最高 46 °C", "転倒 1 回:全速の前進から急に後退へ切り替えたとき"]),
             ]),
             narr="シミュレーションでは、二キロの荷物で転倒はゼロ。実機では十一回立ち上がり、初めての形の荷物を運べました。"),
        dict(kind="split", segs=[dict(src=HB + "model_23984_biped_final_row1_s44_drop10-step-0.mp4", st=0.0, dur=6.0),
                                 dict(src=SIM["drop"], st=0.0, dur=6.0)], min_dur=8.0,
             ov=dict(kicker="考察", title="残った課題", tag=ISAAC, items=[
                 ("腕の間より小さい物は落とす", ["次:挟んで持つ"]),
                 ("止まれの指示でも、少し前後に揺れる", ["次:その場に留まる学習"]),
                 ("立ち続けると、モーターが熱くなる", ["次:両足で止まる姿勢、温度を見て休む"]),
             ]),
             narr="小さい物を挟んで持つこと、その場に留まること、発熱を抑えることが、次の課題です。"),
        dict(kind="still", dur=5.0, im=slide("ありがとうございました", [
            "コード:github.com/korokke-3/go2-biped-carry",
            "使用:Isaac Lab / rsl_rl / MuJoCo / unitree_sdk2",
            "参考にした公開実装:go2_quad2hand、BSD-3 ライセンス",
            f"ナレーション:VOICEVOX:{SPEAKER_NAME}",
            "倍速の場面は、画面右上に倍率を表示",
        ]),
             narr="ご覧いただき、ありがとうございました。"),
    ]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--real", required=True, help="phone recording of the robot")
    p.add_argument("--mujoco", required=True, help="sim2sim_mujoco.py --video clip (stand, walk, turn, go down)")
    p.add_argument("--out", default=os.path.join(ROOT, "outputs", "final_video.mp4"))
    p.add_argument("--tts_cache", default=os.path.join(ROOT, "outputs", "narration"))
    p.add_argument("--preview", default=None, help="only save every slide / overlay as PNG into this directory")
    a = p.parse_args()
    parts = build_parts(a.real, a.mujoco)
    if a.preview:
        os.makedirs(a.preview, exist_ok=True)
        for i, pt in enumerate(parts):
            if pt["kind"] == "still":
                pt["im"].save(os.path.join(a.preview, f"p{i:02d}.png"))
                continue
            if pt["kind"] == "split":
                out = os.path.join(a.preview, f"p{i:02d}.mp4")
                render_split(pt["segs"], 1.0, out, split_overlay(**pt["ov"]), a.preview)
                run([FF, "-y", "-ss", "0.6", "-i", out, "-frames:v", "1", os.path.join(a.preview, f"p{i:02d}.png")])
                os.remove(out)
                continue
            src = pt["src"] if os.path.isabs(pt["src"]) else os.path.join(ROOT, pt["src"])
            fr = os.path.join(a.preview, f"p{i:02d}_frame.png")
            run([FF, "-y", "-ss", str(pt["st"] + pt["dur"] / 2), "-i", src, "-frames:v", "1", "-vf",
                 fit_filter(pt.get("rotate", False)), fr])
            Image.alpha_composite(Image.open(fr).convert("RGBA"), overlay(**pt["ov"], speed=pt.get("speed"))).convert(
                "RGB").save(os.path.join(a.preview, f"p{i:02d}.png"))
            os.remove(fr)
        return
    os.makedirs(a.tts_cache, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        vids, auds, total = [], [], 0.0
        for i, pt in enumerate(parts):
            wavp = os.path.join(a.tts_cache, f"n{i:02d}_{hashlib.md5(pt['narr'].encode()).hexdigest()[:8]}.wav")
            tts(pt["narr"], wavp)
            need = LEAD + wav_len(wavp) + TAIL
            out = os.path.join(tmp, f"p{i:02d}.mp4")
            if pt["kind"] == "still":
                L = render_still(pt["im"], max(pt["dur"], need), out, tmp)
            elif pt["kind"] == "split":
                L = render_split(pt["segs"], need, out, split_overlay(**pt["ov"]), tmp, pt.get("min_dur", 6.0))
            else:
                speed = pt.get("speed", 1.0)
                if pt["dur"] / speed < need and speed > 1.0:
                    speed = max(1.0, round(pt["dur"] / need, 2))  # slow down towards 1x first
                hold = max(0.0, need - pt["dur"] / speed)
                src = pt["src"] if os.path.isabs(pt["src"]) else os.path.join(ROOT, pt["src"])
                ov = overlay(**pt["ov"], speed=speed if speed > 1.0 else None)
                L = render_clip(src, pt["st"], pt["dur"], out, ov, pt.get("rotate", False), speed, hold, tmp)
            # narration track for this part: LEAD s silence, speech, silence up to L
            aout = os.path.join(tmp, f"a{i:02d}.wav")
            run([FF, "-y", "-i", wavp, "-af", f"adelay={int(LEAD * 1000)}:all=1,apad", "-t", f"{L:.3f}",
                 "-ar", "48000", "-ac", "1", aout])
            vids.append(out)
            auds.append(aout)
            total += L
            print(f"part {i:2d}: {L:5.1f} s (narration {wav_len(wavp):4.1f} s) total {total:6.1f} s", flush=True)
        vl, al = os.path.join(tmp, "v.txt"), os.path.join(tmp, "a.txt")
        with open(vl, "w") as f:
            f.writelines(f"file '{x}'\n" for x in vids)
        with open(al, "w") as f:
            f.writelines(f"file '{x}'\n" for x in auds)
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        run([FF, "-y", "-f", "concat", "-safe", "0", "-i", vl, "-f", "concat", "-safe", "0", "-i", al,
             "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest",
             "-movflags", "+faststart", a.out])
    print(f"wrote {a.out} ({total:.1f} s)")


if __name__ == "__main__":
    main()
