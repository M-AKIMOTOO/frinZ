#!/usr/bin/env python3
# AKIMOTO
HISTORY = \
""" **********************
 The history
  2022/06/10
  2022/06/11 update1
  2022/06/12 update2
  2022/06/14 update3
  2022/06/16 update4
  2022/06/17 update5
  2022/06/19 update6
  2022/06/20 update7
  2022/06/21 update8
  2022/07/09 update9
  2022/07/10 update10
  2022/07/10 update11
  2022/07/11 update12
  2022/07/12 update13
  2022/07/13 update14
  2022/07/15 update15
  2022/07/18 update16
  2022/07/22 update17
  2022/07/23 update18
  2022/08/22 update19
  2022/09/28 update20
  2023/01/17 update21
  2023/01/24 update22
  2023/02/28 update23

  Made by AKIMOTO on 2022/06/10
********************** """

VERSION = \
"""+++ version +++
 version 1.0 (2022/07/12): とりあえず完成した（S/N の評価はまだ）．
 version 1.1 (2022/07/13): numba (jit) を使用しなくてもいいようにした．
 version 1.2 (2022/07/15): 積分時間を累積することにより，積分時間 vs S/N のグラフを出力できるようにした．
 version 1.3 (2022/07/18): フリンジの３次元グラフを html で保存してグラフを動かせるようにした．
 version 1.4 (2022/07/22): 32m & 34m が観測する天体の方向を az-el で出力できるようにした．
 version 1.5 (2022/07/22): 解析したデータの周波数をファイル名に組み込んだ．
 version 1.6 (2022/08/22): Delay-Rate 平面の表示範囲を変更して，fringe と同じ表示範囲にして比較しやすいようにした．
 version 1.7 (2022/09/28): FFT 点数が 1024 でないとき，正しく RFI カットをすることができなかったので，修正した．
 version 1.8 (2023/01/17): version 1.7 以前から残っていた，複数範囲の RFI カットのバグを修正．これまでは --rfi 1,50 250,300 としても 250-300 MHz しかカットされなかった．ついでにシェバンも修正．
 version 1.9 (2023/01/24): 積分時間が短いときの delay-rate 平面のカラーマップの表示範囲がおかしいので修正。
 version 1.10 (2023/02/28): 積分時間を累積してフリンジ出力するときに --length と --loop を --cumulate と同時に指定しないとダメだったが，修正して --cumulate だけで可能にした．またノイズレベルが累積した積分時間に応じてルートで減少することを確かめるグラフを出力するようにした．ついでにグラフの目盛りを上下左右に表示されるようにした．
+++"""

import os, sys, binascii, struct, datetime, argparse, math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from scipy.optimize import curve_fit
from astropy import units as u
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time


#plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["xtick.direction"]     = "in"       
plt.rcParams["ytick.direction"]     = "in"       
plt.rcParams["xtick.minor.visible"] = True       
plt.rcParams["ytick.minor.visible"] = True
plt.rcParams["xtick.top"]           = True
plt.rcParams["xtick.bottom"]        = True
plt.rcParams["ytick.left"]          = True
plt.rcParams["ytick.right"]         = True  
plt.rcParams["xtick.major.size"]    = 5          
plt.rcParams["ytick.major.size"]    = 5          
plt.rcParams["xtick.minor.size"]    = 3          
plt.rcParams["ytick.minor.size"]    = 3          
plt.rcParams["axes.grid"]           = False
plt.rcParams["grid.color"]          = "lightgray"
plt.rcParams["axes.labelsize"]      = 15
plt.rcParams["font.size"]           = 12


fname = os.path.basename(sys.argv[0]) # fname is "f"ile name
DETAIL = \
"""
# -------------------------- #
#  プログラムの使用例を示す．#
# -------------------------- #
以下にプログラムの使用例を示す．例で使用している引数は，
個々の事例に細分化して指定しているだけなので，
それぞれの引数を組み合わせて使用することができる．

1. cor ファイルのフリンジやスペクトルの出力結果だけを見たいとき
---------------------------------------------------------------
    Time: %s --input YAMAGU32_YAMAGU34_2022001020304_all.cor
    Freq: %s --input YAMAGU32_YAMAGU34_2022001020304_all.cor --frequency

　周波数領域の結果は引数に --frequency を与えないと出力されない．

2. フリンジとやスペクトルの結果をテキストで出力したいとき
---------------------------------------------------------
　1. の例の末尾に --output を追加するだけ．ただし，オプションの順番は問わない．
    Time: %s --input YAMAGU32_YAMAGU34_2022001020304_all.cor --output
    Freq: %s --input YAMAGU32_YAMAGU34_2022001020304_all.cor --frequency --output

3. フリンジとスペクトルのグラフを出力したいとき
-----------------------------------------------
　こちらも 2. と同様に 1. の末尾に --plot を追加するだけ．
　併せてテキストにも出力したいなら，--output も追加する．
    Time: %s --input YAMAGU32_YAMAGU34_2022001020304_all.cor --plot
    Freq: %s --input YAMAGU32_YAMAGU34_2022001020304_all.cor --frequency --plot

　次から説明する --length や --loop を併せて用いることで，
　指定した積分時間ごとでもグラフを出力できる．

4. 積分時間を変えてフリンジやスペクトルを出力したいとき
-------------------------------------------------------
　--length を引数に追加する．例として積分時間１０秒で出力したいとき
    Time: %s --input YAMAGU32_YAMAGU34_2022001020304_all.cor --length 10
    Freq: %s --input YAMAGU32_YAMAGU34_2022001020304_all.cor --frequency --length 10
    
　しかし，これでは観測開始時刻から１０秒のフリンジやスペクトルの出力しかしない．
　観測時間を --length で指定した積分時間で区切って出力したいときは，次の --loop を追加する．

5. 指定した積分時間ごとにフリンジやスペクトルの出力を見たいとき
---------------------------------------------------------------
　4. の例では，cor ファイルに記述されている観測開始時間から１０秒分の
　フリンジやスペクトルしか見れない．指定した積分時間でフリンジやスペクトル
　を出力したいときは，4. の例に --loop を追加する．
    Time: %s --input YAMAGU32_YAMAGU34_2022001020304_all.cor --length 10 --loop 20
    Freq: %s --input YAMAGU32_YAMAGU34_2022001020304_all.cor --frequency --length 10 --loop 20

　上記の例では，積分時間１０秒でフリンジやスペクトルを計算し，それを２０回
　繰り返す，つまり観測開始時刻から２００秒間のデータを１０秒間隔でそれらを
　出力する．

6. フリンジやスペクトルを出力する開始時刻をずらしたいとき
---------------------------------------------------------
　4. や 5. では，cor ファイルに記述されている観測開始時刻から指定した積分時間分
　だけの出力を行っている．観測開始時刻から１７秒だけずらしてからフリンジやスペクトル
　を出力したいときは --skip を追加する．
    Time: %s --input YAMAGU32_YAMAGU34_2022001020304_all.cor --length 10 --loop 20 --skip 17
    Freq: %s --input YAMAGU32_YAMAGU34_2022001020304_all.cor --frequency --length 10 --loop 20 --skip 17

　2. でも述べているが，--length や --loop，--skip の順序は問わない．しかし，
　それぞれに指定する値が異なると，違った出力になる．

7. RFI カットをしたいとき
-------------------------
　こちらも 2. と同様に 1. の末尾に --rfi を追加するだけ．
　RFI をカットしたい周波数の範囲を 1-150 MHz とすると
    Time: %s --input YAMAGU32_YAMAGU34_2022001020304_all.cor --rfi 1,150
    Freq: %s --input YAMAGU32_YAMAGU34_2022001020304_all.cor --frequency --rfi 1,150

　他にも 300-450 MHz もカットしたいときは
    Time: %s --input YAMAGU32_YAMAGU34_2022001020304_all.cor --rfi 1,150 300,450
    Freq: %s --input YAMAGU32_YAMAGU34_2022001020304_all.cor --frequency --rfi 1,150 300,450
    
　1,150 と 300,450 は順不同である．

8. 積分時間を累積して，積分時間に対する S/N のグラフを出力したいとき
------------------------------------------------------------------
　引数 --cumulate を指定する
    Time: %s --input YAMAGU32_YAMAGU34_2022001020304_all.cor --cumulate 1

　とすると，積分時間１秒で１秒ごと（１秒，２秒，３秒...）にフリンジを計算する．

9. ダイナミックスペクトルを出力したいとき
-----------------------------------------
　引数に --dynamic-spectrum を追加する．ダイナミックスペクトルの出力では --length や --loop，--skip を
　指定しても無視される．また --frequency の有無にも関わらず出力する．
    Time: %s --input YAMAGU32_YAMAGU34_2022001020304_all.cor --dynamic-spectrum
    Freq: %s --input YAMAGU32_YAMAGU34_2022001020304_all.cor --frequency --dynamic-spectrum

　上記のどちらにしても同じダイナミックスペクトルを得ることができる．

10. Delay-Rate サーチ平面の３次元プロットを出力したいとき
--------------------------------------------------------
    Time: %s --input YAMAGU32_YAMAGU34_2022001020304_all.cor --3D
    Freq: %s --input YAMAGU32_YAMAGU34_2022001020304_all.cor --frequency --3D

　html ファイルで保存するので，様々な角度からフリンジを確認することができる．
　ダイナミックスペクトルを出力するときと同様に，上記のどちらにしても同じ図を得ることができる．
　ただし，2022/07/11 時点ではカラーバーの設定が改善されていない（2022/07/18 に改善した）．

11. Frequency-Rate の２次元データ，Delay-Rate の２次元データを出力したいとき
----------------------------------------------------------------------------
　引数に --cross-output を指定する．
    Time: %s --input YAMAGU32_YAMAGU34_2022001020304_all.cor --cross-output
    Freq: %s --input YAMAGU32_YAMAGU34_2022001020304_all.cor --frequency --cross-output

以上（2022/07/11 AKIMOTO）""" % (fname,fname,fname,fname,fname,fname,fname,fname,fname,fname,fname,fname,fname,fname,fname,fname,fname,fname,fname,fname,fname,fname,fname)

DESCRIPTION = \
"""
# ---------- #
# 簡単な説明 #
# ---------- #
　%s は fringe コマンドの補助となるプログラムである．fringe は Delay-Rate サーチ平面でピーク値を持っているが，
　これでは混信が見られるデータでは，こちらを検出してしまう．山口干渉計では，キャリブレーターを用いて，きちんと
　遅延時間を決定すれば，混信の影響を受けずに (Delay, Rate) = (0, 0) に天体の信号が現れる（検出できれば...）．
　このプログラムでは，サーチ内のピークではなく，(Delay, Rate) = (0, 0) の信号を必ず取ってくる．こうすれば，
　混信の影響を受けずに天体信号を取り出すことができるので，fringe コマンドで行っていた RFI のカットをしなくて
　よくなる．しかし，(Delay, Rate) = (0, 0) でも混信が見られるなら RFI カットの必要はある．

# ---------- #
# 引数の詳細 #
# ---------- #
　%s の詳細な使い方は \"%s --detail\" を実行することで確認できる．
""" % (fname, fname, fname)

EPILOG = \
"""\n
# ---------- #
# 修正や追加 #
# ---------- #
　プログラムのバグや処理の追加などがありましたら，穐本（c001wbw[at]yamaguchi-u.ac.jp，[at] を @ へ変更してください）まで
　連絡してください．連絡先は 2022/07/11 時点のものです．
"""
# arguments
class MyHelpFormatter(argparse.ArgumentDefaultsHelpFormatter,
                      argparse.RawDescriptionHelpFormatter):
    pass
parser = argparse.ArgumentParser(description=DESCRIPTION, epilog=EPILOG, formatter_class=MyHelpFormatter)  

help_input     = "cor ファイルを指定する（１つまで）．"
help_length    = "積分時間（PP単位，基本的には秒単位）を指定する．"
help_skip      = "cor ファイル中の観測開始時刻から，どれくらいの時間を飛ばすか．"
help_loop      = "引数 --length で指定した積分時間分のフリンジ出力を何回繰り返すか．"
help_output    = "フリンジもしくはスペクトルの出力をテキストファイルとして出力する．"
help_plot      = "フリンジのグラフを出力する．スペクトルのグラフを出力したいときは，引数 --frequency と同時に使用する．"
help_freq      = "スペクトルのグラフを出力する．引数 --plot と同時に使用する．"
help_rfi       = "周波数カット．切り取りたい周波数の最小値と最大値を次のように指定する．--rfi freq_min,freq_max．複数の周波数カットをしたい場合は --rfi freq_min1,freq_max1 freq_min2,freq_max2 ... とする．"
help_cumulate  = "指定した時間間隔で積分時間を累積していき，積分時間 vs S/N の両対数グラフを出力する．また積分時間 vs ノイズレベルのグラフも出力する．"
help_cross     = "Frequency-Rate の２次元データ，Delay-Rate の２次元データを csv ファイルとして出力する．同時に cor ファイルのデータも出力する．"
help_dynamic   = "ダイナミックスペクトルとフリンジの時間変化を出力する．混信の把握に便利．"
help_3D        = "Delay-Rate サーチ平面の３次元プロットを表示する．２パターンの画像が出力され，１つはすべてのサーチ平面をプロットしており，１つは (Delay,Rate) = (0,0) の部分を拡大したサーチ平面をプロットしている．"
help_add_plot  = "相関振幅，位相，S/N の時間変化をプロットする．"
help_summarize = "すべての出力ファイルをディレクトリにまとめる．"
help_history   = "このプログラムの変更履歴を表示する．"
help_version   = "このプログラムのバージョンを表示する．"
help_detail    = "このプログラムの使い方の詳細を表示する．"

parser.add_argument("--input"           , default=True , type=str            , help=help_input    )
parser.add_argument("--length"          , default=0    , type=int            , help=help_length   )
parser.add_argument("--skip"            , default=0    , type=int            , help=help_skip     )
parser.add_argument("--loop"            , default=False, type=int            , help=help_loop     )
parser.add_argument("--output"          , action="store_true"                , help=help_output   )
parser.add_argument("--plot"            , action="store_true"                , help=help_plot     )
parser.add_argument("--frequency"       , action="store_true"                , help=help_freq     )
parser.add_argument("--rfi"             , default=False, nargs="*"           , help=help_rfi      )
parser.add_argument("--cumulate"        , default=0    , type=int            , help=help_cumulate )
parser.add_argument("--cross-output"    , action="store_true", dest="cross"  , help=help_cross    )
parser.add_argument("--dynamic-spectrum", action="store_true", dest="dynamic", help=help_dynamic  )
parser.add_argument("--3D"              , action="store_true", dest="ddd"    , help=help_3D       )
parser.add_argument("--add-plot"        , action="store_true", dest="addplot", help=help_add_plot )
parser.add_argument("--summarize"       , action="store_true"                , help=help_summarize)
parser.add_argument("--history"         , action="store_true"                , help=help_history  )
parser.add_argument("--version"         , action="store_true"                , help=help_version  )
parser.add_argument("--detail"          , action="store_true"                , help=help_detail   )

args = parser.parse_args() 
ifile     = args.input
length    = args.length
skip      = args.skip
loop      = args.loop
output    = args.output
time_plot = args.plot
freq_plot = args.frequency
rfi       = args.rfi
cumulate  = args.cumulate
cs_output = args.cross # cs: "c"ross-"s"pectrum
add_plot  = args.addplot
DDD       = args.ddd   # 2D-graph
summarize = args.summarize
ds_plot   = args.dynamic
history   = args.history
version   = args.version
detail    = args.detail

def help_show() :
    os.system("%s --help" % os.path.basename(sys.argv[0]))
    exit(0)
if len(sys.argv) == 1 or not ifile :
    help_show()
if history == True :
    print(HISTORY) ; exit(0)
if detail == True  :
    print(DETAIL)  ; exit(0)
if version == True :
    print(VERSION) ; exit(0)

def hex_to_string(conv: str) -> str:
    conv_string   = bytes.fromhex(conv).decode("utf-8")
    return conv_string

def hex_to_integer(conv: str) -> int :
    conv_integer = int("%s" % conv, 16)
    return conv_integer

def hex_to_float(conv: str) -> float :
    conv_float = struct.unpack('!f', bytes.fromhex(conv))
    return conv_float

def hex_to_double(conv: str) -> float :
    conv_double = struct.unpack('>d', binascii.unhexlify(conv))
    return conv_double

def cor_read_to_hex(byte: str) -> str :
    cor_file = open(ifile, "rb")
    cor_byte_string = ""
    cor_read_region = cor_file.read(byte)
    cor_read_buffer = np.frombuffer(cor_read_region, dtype=">S1")
    for cor_buff in cor_read_buffer :
        cor_hex = cor_buff.hex()
        if not cor_hex :
            cor_byte_string += "00"
        cor_byte_string += cor_hex
    
    one_line_characters = 32
    cor_line = int(len(cor_byte_string) / one_line_characters)
    cor_file.close()
    return cor_line, cor_byte_string

def address_split(address_data: str) -> str :
    address00_sp = address_data[0:2] # 2 byte, the same as beloww
    address01_sp = address_data[2:4]
    address02_sp = address_data[4:6]
    address03_sp = address_data[6:8]
    address04_sp = address_data[8:10]
    address05_sp = address_data[10:12]
    address06_sp = address_data[12:14]
    address07_sp = address_data[14:16]
    address08_sp = address_data[16:18]
    address09_sp = address_data[18:20]
    address0a_sp = address_data[20:22]
    address0b_sp = address_data[22:24]
    address0c_sp = address_data[24:26]
    address0d_sp = address_data[26:28]
    address0e_sp = address_data[28:30]
    address0f_sp = address_data[30:32]

    return address00_sp, address01_sp, address02_sp, address03_sp, address04_sp, address05_sp, address06_sp, address07_sp, \
        address08_sp, address09_sp, address0a_sp, address0b_sp, address0c_sp, address0d_sp, address0e_sp, address0f_sp

def Radian2RaDec(RA_radian: float, Dec_radian: float) -> float :
    Ra_deg  = math.degrees(RA_radian)
    Dec_deg = math.degrees(Dec_radian)
    return Ra_deg, Dec_deg

def RaDec2AltAz(object_ra: float, object_dec: float, observation_time: float, latitude: float, longitude: float, height: float) -> float :
    location_geocentrice = EarthLocation.from_geocentric(latitude, longitude, height, unit=u.m)
    location_geodetic    = EarthLocation.to_geodetic(location_geocentrice)
    location_lon_lat     = EarthLocation(lon=location_geodetic.lon, lat=location_geodetic.lat, height=location_geodetic.height)
    obstime              = Time("%s" % observation_time)
    object_ra_dec        = SkyCoord(ra=object_ra*u.deg, dec=object_dec*u.deg)
    AltAz_coord          = AltAz(location=location_lon_lat, obstime=obstime)
    object_altaz         = object_ra_dec.transform_to(AltAz_coord)
    
    return object_altaz.az.deg, object_altaz.alt.deg, location_lon_lat.height.value

def zerofill(integ: int) -> int :
    powers_of_two = 1
    integ_re      = integ
    while True :
        powers_of_two = powers_of_two * 2
        integ = integ / 2
        if integ < 1.0 :
            break
        else :
            continue
    zero_num = int(powers_of_two - integ_re)
    return powers_of_two, zero_num

def RFI(r0, bw: int, fft: int) -> int :
    # RFI
    rfi_cut = []
    for r1 in r0 :
        rfi_range = r1.split(",")
        rfi_min = int(rfi_range[0])
        rfi_max = int(rfi_range[1])
        if rfi_max > 512 :
            rfi_max = 512
        if rfi_min < 0 or rfi_max < 0 :
            print("The RFI minimum, %.0f, or maximum, %.0f, frequency is more than 0." % (rfi_min, rfi_max))
            exit(1)
        elif rfi_min >= rfi_max :
            print("The RFI maximum frequency, %.0f, is smaller than the RFI minimum frequency, %.0f." % (rfi_min, rfi_max))
            exit(1)
        else :
            pass

        r2 = int(rfi_min) * int(fft/2/bw); rfi_cut.append(r2)
        r3 = int(rfi_max) * int(fft/2/bw); rfi_cut.append(r3)
        
        

    return rfi_cut

#from numba import jit
#@jit
def noise_level(input_2D_data: float, search_00_amp: float) -> float :
    
    input_2D_data_real_imag_ave = np.mean(input_2D_data) # 複素数でも実部と虚部でそれぞれで平均を計算できるみたい．
    noise_level = np.mean(np.absolute(input_2D_data - input_2D_data_real_imag_ave)) # 信号の平均値が直流成分に対応するため，それを除去のために平均値を引いている？．加算平均をとることで雑音レベルを下げることができるらしい．
    
    try :
        SNR = search_00_amp / noise_level
    except ZeroDivisionError :
        SNR, noise_level = 0.0, 0.0

    return SNR, noise_level

def time_add_plot(x1, y1, xl, yl, s, Xm, xM) :
    fig = plt.figure(figsize=(9,6))
    plt.plot(x1, y1, "o")
    plt.xlabel("The elapsed time since %s UT" % xl)
    plt.ylabel("%s" % yl)
    plt.xlim([Xm, xM])
    if s == "snr" :
        plt.ylim(ymin=0)
    elif s == "phase" :
        plt.yticks([-180, -120, -60, 0, 60, 120, 180])
    plt.tight_layout()
    plt.savefig("%s/%s_add_plot_%s.png" % (save_file_pass, save_file_name, s))
    #plt.show()
    plt.clf(); plt.close()


def dynamic_spectrum(x,y,z,p,n) :
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    c = ax.contourf(x, y, np.absolute(z), 100, cmap="rainbow")
    if   n == 1 : # the frequency domain
        xlabel     = "frequency [MHz]"
        ylabel     = "length [s]"
        zlabel     = "cross-spectrum [a.u.]"
        save_name  = "%s/%s_dynamic_spectrum_frequency.png" %  (save_file_pass, save_file_name)
        xmin, xmax = 0, 512
        ymin, ymax = 0, p
    elif n == 2 : # the time domain
        xlabel     = "time lag [samles]"
        ylabel     = "length [s]"
        zlabel     = "cross-amplitude [a.u.]"
        save_name  = "%s/%s_dynamic_spectrum_time_lag.png" %  (save_file_pass, save_file_name)
        xmin, xmax = -fft_point//2, fft_point//2
        ymin, ymax = 0, p
    fig.colorbar(c, label=zlabel)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])
    plt.grid()
    plt.tight_layout()
    plt.savefig(save_name)
    plt.clf(); plt.close()

# label
label = os.path.splitext(ifile)[0].split("_")[-1]

#
# the directory to save the output data
#
save_directory_pass = os.path.dirname(ifile)
if save_directory_pass == "" :
    save_directory_pass = "./frinZ"
else :
    save_directory_pass += "/frinZ"
os.makedirs(save_directory_pass, exist_ok=True)

#
# header region
#
cor_header_file = "%s/%s_header.txt" % (save_directory_pass, os.path.basename(ifile).split(".")[0])
cor_header_file_exist = os.path.isfile(cor_header_file)
cor_line_num, cor_byte_conv = cor_read_to_hex(256)
for j in range(cor_line_num) :
    # Address line of cor-file format
    address = cor_byte_conv[j*32:32*(j+1)]

    # each address data
    address00, address01, address02, address03, address04, address05, address06, address07, \
        address08, address09, address0a, address0b, address0c, address0d, address0e, address0f = address_split(address)
    
    address_00_01_02_03 = "%s%s%s%s" % (address00, address01, address02, address03) # 8 byte, the same as below
    address_04_05_06_07 = "%s%s%s%s" % (address04, address05, address06, address07)
    address_03_02_01_00 = "%s%s%s%s" % (address03, address02, address01, address00) 
    address_07_06_05_04 = "%s%s%s%s" % (address07, address06, address05, address04)
    address_0b_0a_09_0a = "%s%s%s%s" % (address0b, address0a, address09, address08)
    address_0f_0e_0d_0c = "%s%s%s%s" % (address0f, address0e, address0d, address0c)

    if   j == 0 :
        magic_word          = address_03_02_01_00
        header_version      = hex_to_integer(address_07_06_05_04)
        software_version    = hex_to_integer(address_0b_0a_09_0a)
        sampling_speed      = hex_to_integer(address_0f_0e_0d_0c) / 10**6
    elif j == 1 :                                                                           # FFT point, Number of Sector, observing frequency and parametr period
        fft_point           = hex_to_integer(address_0b_0a_09_0a)
        number_of_sector    = hex_to_integer(address_0f_0e_0d_0c)
        observing_frequency = hex_to_double(address_07_06_05_04 + address_03_02_01_00)[0] / 10**6
        PP  = number_of_sector         # parameter period
        BW  = int(sampling_speed // 2) # bandwidth in Yamaguchi-Univ.
        RBW = BW / (fft_point // 2)    # resolution bandwidth
    elif j == 2 :                                                                           # Station1-Name, It is namely the used antenna1
        station1_name       = hex_to_string(address_00_01_02_03 + address_04_05_06_07)
    elif j == 3 :
        station1_position_x = hex_to_double(address_07_06_05_04 + address_03_02_01_00)[0]   # return the tuple
        station1_position_y = hex_to_double(address_0f_0e_0d_0c + address_0b_0a_09_0a)[0]   # return the tuple
    elif j == 4 :
        station1_position_z = hex_to_double(address_07_06_05_04 + address_03_02_01_00)[0]   # return the tuple
        station1_code       = hex_to_string(address08)
    elif j == 5 :                                                                           # Station1-Name, It is namely the used antenna2
        station2_name       = hex_to_string(address_00_01_02_03 + address_04_05_06_07)
    elif j == 6 :
        station2_position_x = hex_to_double(address_07_06_05_04 + address_03_02_01_00)[0]   # return the tuple
        station2_position_y = hex_to_double(address_0f_0e_0d_0c + address_0b_0a_09_0a)[0]   # return the tuple
    elif j == 7 :
        station2_position_z = hex_to_double(address_07_06_05_04 + address_03_02_01_00)[0]   # return the tuple
        station2_code       = hex_to_string(address08)
    elif j == 8 :                                                                           # Source-Name
        address_replace     = (address_00_01_02_03 + address_04_05_06_07).replace("00", "")
        source_name         = hex_to_string(address_replace)
    elif j == 9 :
        source_position_ra  = hex_to_double(address_07_06_05_04 + address_03_02_01_00)[0]; source_position_ra  = Radian2RaDec(source_position_ra, 0)[0]   # return the tuple
        source_position_dec = hex_to_double(address_0f_0e_0d_0c + address_0b_0a_09_0a)[0]; source_position_dec = Radian2RaDec(0, source_position_dec)[1]   # return the tuple
    elif j == 13 :
        total_delay = hex_to_double(address_0f_0e_0d_0c + address_0b_0a_09_0a)[0]

#       
# Header Region Information
#
if cor_header_file_exist == False :
    header_region_info = \
    """##### Header Region
    sampling frequency = %.0f MHz
    Magic Word = %s
    Sofrware Vesion = %.0f
    Header Version = %.0f
    Observing Frequency = %.0f MHz
    FFT Point = %.0f
    PP (parameter period) = %.0f
    BandWidth = %.0f MHz
    Resolution BandWidth = %.5f MHz
    total delay = %s s

    Station1
    Name = %s
    Code = %s
    Pisition (X,Y,Z) = (%f,%f,%f) m, geocentric coordinate
    Station2
    Name = %s
    Code = %s
    Pisition (X,Y,Z) = (%f,%f,%f) m, geocentric coordinate

    Source
    Name = %s
    Position (RA, Decl) = (%f,%f) deg, J2000
    """ % (sampling_speed, magic_word, software_version, header_version, observing_frequency, fft_point, number_of_sector, BW, RBW, total_delay, \
        station1_name, station1_code, station1_position_x, station1_position_y, station1_position_z, \
        station2_name, station2_code, station2_position_x, station2_position_y, station2_position_z, \
        source_name, source_position_ra, source_position_dec)
    cor_header_save = open(cor_header_file, "w"); print(header_region_info, file=cor_header_save)

#
# correlation processing data secotor
#
data_sector_region_byte = 16 * fft_point / 2 * number_of_sector # the cor-file structure
each_sector_header_byte = 16 * 8 * number_of_sector 
total_byte = int(data_sector_region_byte + each_sector_header_byte) + 256
cor_sector_line_num, cor_sector_byte_conv = cor_read_to_hex(byte=total_byte)

sector_start = [int((8+fft_point/2**2)*(pp)+16)  for pp in range(number_of_sector)]
sector_end   = [int((8+fft_point/2**2)*(pp+1)+16)  for pp in range(number_of_sector)]

#
# adjust the length, the skip and the integration time
#
# check the integration time
if (PP-skip) < length :
    print("You sepcify the integraion time %.2f seconds and the skip time %.2f seconds." % (length, skip))
    print("The observation time, however, is %.2f seconds." % (PP))
    print("Please execute \"%s --detail\"." % fname)
    exit(1)

# length
if length < 0 or length == 0 or length > PP :
    length = PP

# skip
if skip < 0 or skip >= PP:
    print("As following the condition, please specify the skip argument.")
    print("# 0 < skip <= %f" % PP)
    print("You specify the skip time, %.0f" % skip)
    exit(1)

# loop
if loop == False or (PP-skip)//length <= 0  :
    loop = 1
elif loop >= (PP-skip)//length :
    loop = (PP-skip)//length

if cumulate >= PP :
    print("The specified cumulation length, %d s, is more than the observation time, %d s." % (cumulate, PP))
    exit(1)
if cumulate != 0 and (add_plot == True or freq_plot == True) :
    print("You can specify the argument whether \"--cumulate\" or \"--add-plot\"")
    print("and \"--cumulate\" can't specify with \"--frequency\".")
    exit(1)
if cumulate != 0 :
    length = 0
    loop = int(PP/cumulate)

# frequency
if   6600 <= observing_frequency <= 7112 :
    observing_band = "c-band"
elif 8192 <= observing_frequency <= 8704 :
    observing_band = "x-band"
else :
    observing_band = ""

complex_spectrum       = "#index\tlen[s]\tfreq[MHz; RBW unit]\treal\timag\n"
integ_complex_2D_array = []; integ_complex_2D_array_append = integ_complex_2D_array.append
corr_start_sec         = []
for sector in range(skip, PP) :

    each_sector_data_address = cor_sector_byte_conv[32*sector_start[sector]:32*sector_end[sector]]

    each_sector_header = each_sector_data_address[:256] # the header of each sector regions
    each_sector_data   = each_sector_data_address[256:] # the data of each sector regions

    #
    # each sector header
    #
    for e in range(8) :

        each_sector_header_line = each_sector_header[32*e:32*(e+1)]

        address00_sector_header, address01_sector_header, address02_sector_header, address03_sector_header, address04_sector_header, address05_sector_header, address06_sector_header, address07_sector_header, \
        address08_sector_header, address09_sector_header, address0a_sector_header, address0b_sector_header, address0c_sector_header, address0d_sector_header, address0e_sector_header, address0f_sector_header = address_split(each_sector_header_line)
        
        address_03_02_01_00_sector_header = "%s%s%s%s" % (address03_sector_header, address02_sector_header, address01_sector_header, address00_sector_header) # 8 byte, the same as below
        address_07_06_05_04_sector_header = "%s%s%s%s" % (address07_sector_header, address06_sector_header, address05_sector_header, address04_sector_header)
        address_0b_0a_09_08_sector_header = "%s%s%s%s" % (address0b_sector_header, address0a_sector_header, address09_sector_header, address08_sector_header)
        address_0f_0e_0d_0c_sector_header = "%s%s%s%s" % (address0f_sector_header, address0e_sector_header, address0d_sector_header, address0c_sector_header)

        if e == 0 :
            correlation_start_sec    = hex_to_integer(address_03_02_01_00_sector_header) # since 1970/01/01 00:00:00 UT
            correlation_stop_sec     = hex_to_integer(address_0b_0a_09_08_sector_header) # since 1970/01/01 00:00:00 UT
        elif e == 1 : # Station-1 Clock Delay, It is namely the delay of antenna1
            station1_clock_epoch_sec = hex_to_integer(address_03_02_01_00_sector_header) # since 1970/01/01 00:00:00 UT
            station1_clock_delay     = hex_to_double(address_0f_0e_0d_0c_sector_header + address_0b_0a_09_08_sector_header)
        elif e == 2 : # Station-1 Clock Rate, Acel
            station1_clock_rate      = hex_to_double(address_07_06_05_04_sector_header + address_03_02_01_00_sector_header)
        #    station1_clock_acel      = hex_to_double(address_0f_0e_0d_0c_sector_header + address_0b_0a_09_08_sector_header)
        #elif e == 3 : # Station-1 Clock Jerk
        #    station1_clock_jerk      = hex_to_double(address_07_06_05_04_sector_header + address_03_02_01_00_sector_header)
        #    station1_clock_snap      = hex_to_double(address_0f_0e_0d_0c_sector_header + address_0b_0a_09_08_sector_header)
        elif e == 4 : # Station-1 Clock Delay, It is namely the delay of antenna1
            station2_clock_epoch_sec = hex_to_integer(address_03_02_01_00_sector_header) # since 1970/01/01 00:00:00 UT
            station2_clock_delay     = hex_to_double(address_0f_0e_0d_0c_sector_header + address_0b_0a_09_08_sector_header)
        elif e == 5 : # Station-1 Clock Rate, Acel
            station2_clock_rate      = hex_to_double(address_07_06_05_04_sector_header + address_03_02_01_00_sector_header)
        #    station2_clock_acel      = hex_to_double(address_0f_0e_0d_0c_sector_header + address_0b_0a_09_08_sector_header)
        #elif e == 6 : # Station-1 Clock Jerk
        #    station2_clock_jerk      = hex_to_double(address_07_06_05_04_sector_header + address_03_02_01_00_sector_header)
        #    station2_clock_snap      = hex_to_double(address_0f_0e_0d_0c_sector_header + address_0b_0a_09_08_sector_header)
        elif e == 7 :
            effective_integration_length = np.round(hex_to_float(address_03_02_01_00_sector_header)[0], 5)
    observation_start_time = datetime.datetime(1970,1,1,0,0,0) + datetime.timedelta(seconds=correlation_start_sec)
    corr_start_sec.append(observation_start_time)

    #
    # each sector data
    #
    f1, f2 = 0, 1 # This value is the frequency of the cross-power spectrum, 0-511 MHz from observing frequency (e.x. 6600, 8192 MHz)
    of1, of2 = observing_frequency, observing_frequency+1 # "of" is "o"bserving "f"requency
    complex_temp = []; complex_temp_append = complex_temp.append
    for x in range(fft_point//2**2) :

        each_sector_data_line = each_sector_data[32*x:32*(x+1)]

        address00_sector_data, address01_sector_data, address02_sector_data, address03_sector_data, address04_sector_data, address05_sector_data, address06_sector_data, address07_sector_data, \
        address08_sector_data, address09_sector_data, address0a_sector_data, address0b_sector_data, address0c_sector_data, address0d_sector_data, address0e_sector_data, address0f_sector_data = address_split(each_sector_data_line)

        address_03_02_01_00_sector_data = "%s%s%s%s" % (address03_sector_data, address02_sector_data, address01_sector_data, address00_sector_data) # 8 byte, the same as below
        address_07_06_05_04_sector_data = "%s%s%s%s" % (address07_sector_data, address06_sector_data, address05_sector_data, address04_sector_data)
        address_0b_0a_09_08_sector_data = "%s%s%s%s" % (address0b_sector_data, address0a_sector_data, address09_sector_data, address08_sector_data)
        address_0f_0e_0d_0c_sector_data = "%s%s%s%s" % (address0f_sector_data, address0e_sector_data, address0d_sector_data, address0c_sector_data)

        cross_spectrum_real1 = hex_to_float(address_03_02_01_00_sector_data)[0]
        cross_spectrum_imag1 = hex_to_float(address_07_06_05_04_sector_data)[0]
        cross_spectrum_real2 = hex_to_float(address_0b_0a_09_08_sector_data)[0]
        cross_spectrum_imag2 = hex_to_float(address_0f_0e_0d_0c_sector_data)[0]
        
        complex_spectrum += "%.0f\t%.0f\t%.0f\t%+.5e\t%+.5e\n%.0f\t%.0f\t%.0f\t%+.5e\t%+.5e\n" % (f1, sector, of1, cross_spectrum_real1, cross_spectrum_imag1, f2, sector, of2, cross_spectrum_real2, cross_spectrum_imag2)

        f1  += 2; f2  += 2; of1 += 2; of2 += 2

        if x == 0 : # fringe コマンドでは，DC成分をカットするために，PP単位ごとの real と imag のデータの一番最初を０としている            
            #cross_spectrum_real1, cross_spectrum_imag1 = 0, 0
            pass

        complex1 = complex(cross_spectrum_real1, cross_spectrum_imag1)
        complex2 = complex(cross_spectrum_real2, cross_spectrum_imag2); complex_temp_append(complex1); complex_temp_append(complex2)
    
    integ_complex_2D_array_append(complex_temp) # 2D array: (rows, columns) = (PP, FFT/2) = (integ time, bandwidth)
    complex_temp_append = [] # release the RAM
    complex_spectrum += "\n"
if cs_output == True :
    cor_output = open("%s/%s_complex_spectrum.tsv" % (save_directory_pass, os.path.basename(ifile).split(".")[0]), "w"); print(complex_spectrum, file=cor_output)


# convert list to ndarray
integ_complex_2D_array = np.array(integ_complex_2D_array)

cumulate_len, cumulate_snr, cumulate_noise = [], [], []
add_plot_length, add_plot_amp, add_plot_snr, add_plot_phase, add_plot_noise_level = [], [], [], [], []
for l in range(loop) :
    save_file_name = ""

    # the cumulation of the integration time.
    if cumulate != 0 :
        if length <= PP :
            length += cumulate
            l = 0
    if length > PP : # for --cumulate
        break

    # epoch
    epoch0 = corr_start_sec[length*l:length*(l+1)]
    epoch1 = "%s" % epoch0[0].strftime("%Y/%j %H:%M:%S")
    epoch2 = "%s" % epoch0[0].strftime("%Y%j%H%M%S")
    epoch3 = "%s" % (epoch0[0] + datetime.timedelta(seconds=PP)).strftime("%Y-%m-%d %H:%M:%S")

    # azel 
    station1_azel = RaDec2AltAz(source_position_ra, source_position_dec, epoch3, station1_position_x, station1_position_y, station1_position_z)
    station2_azel = RaDec2AltAz(source_position_ra, source_position_dec, epoch3, station2_position_x, station2_position_y, station2_position_z)

    # the directory to summarize the output file
    if summarize == True and l == 0 :
        save_directory_pass += "/%s/len%d" % (epoch2, length)
        os.makedirs(save_directory_pass, exist_ok=True)

    # save-file name
    if cumulate == 0 or (cumulate != 0 and l == 0) :
        save_file_name += "%s_%s_%s_%s_%s_len%.0fs" % (station1_name, station2_name, epoch2, label, observing_band, length)
    if rfi != False:
        save_file_name += "_rfi"
    if cumulate != 0 :
        save_file_name += "_cumulate%ds" % cumulate
        
    save_file_pass = save_directory_pass

    # convert list to ndarray
    integ_complex_2D_array_split = integ_complex_2D_array[length*l:length*(l+1)]
    
    #
    # IFFT & FFT
    #
    freq_rate_2D_array = []; freq_rate_2D_array_append = freq_rate_2D_array.append
    integ_fft = 4 *zerofill(integ=length)[0] # the FFT in the time (same as integration time) direction. rate
    for i in range(fft_point//2) :
        # RFI cut
        if rfi != False :
            RFI_cut = RFI(r0=rfi, bw=BW, fft=fft_point)
            for r in range(0, len(RFI_cut), 2) :
                if RFI_cut[r] <= i <= RFI_cut[r+1] :
                    integ_complex_2D_array_split[:,i] = complex(0,0)
                else :
                    pass    

        fft_time_direction = np.fft.fft(integ_complex_2D_array_split[:,i], n=integ_fft) * fft_point / length
        freq_rate_2D_array_append(fft_time_direction) # frequency vs rate time in th dynamic spectrum

    freq_rate_2D_array = np.array(freq_rate_2D_array) # 2D array: (rows, columns) = (FFT/2, rate), the number of "rate" is equal to PP.
    freq_rate_2D_array = np.concatenate([freq_rate_2D_array[:,integ_fft//2:], freq_rate_2D_array[:,:integ_fft//2]], 1)

    lag_rate_2D_array = []; lag_rate_2D_array_append = lag_rate_2D_array.append

    for i in range(integ_fft) :
        ifft_freq_direction = np.fft.ifft(freq_rate_2D_array[:,i], n=fft_point)
        lag_rate_2D_array_append(ifft_freq_direction)

    lag_rate_2D_array = np.array(lag_rate_2D_array)
    #lag_rate_2D_array = np.concatenate([lag_rate_2D_array[PP//2:], lag_rate_2D_array[:PP//2]])                           # rate 方向でデータの半分を左右で入れ替える．
    lag_rate_2D_array = np.concatenate([lag_rate_2D_array[:, fft_point//2+1:], lag_rate_2D_array[:, :fft_point//2+1]], 1) # 周波数方向でデータ半分を入れ替える．
    lag_rate_2D_array = lag_rate_2D_array[:, ::-1]                                                                        # 列反転，これは delay が０を中心に対称になるため．

    #
    # the cross-spectrum, the fringe phase, the rate in the frequency domain, the time-lag, and the rate in the time domain.
    #
    integ_range = np.round(np.linspace(0,PP,PP), 5)                                             # integration time range
    rate_range  = np.fft.fftshift(np.fft.fftfreq(integ_fft, d=effective_integration_length))    # rate range, the sampling frequency is 1 second if the outout value in xml-file is 1 Hz and the parameter if length is 1.
    freq_range  = np.round(np.linspace(0,512,int(512/RBW)),3)                                   # cross spectrum range
    lag_range   = np.round(np.linspace(-fft_point//2,fft_point//2-1,fft_point), 5)              # time lag range

    yi_time_lag  = 0.0
    yi_time_rate = 0.0
    yi_freq_rate = 0.0

    #
    # fringe search
    #
    # frequency domain
    if freq_plot == True :
        fringe_freq_rate_00_complex_index = np.where(rate_range==yi_freq_rate)[0][0]
        fringe_freq_rate_00_spectrum      = np.absolute(freq_rate_2D_array[:,fringe_freq_rate_00_complex_index])
        fringe_freq_rate_00_phase1        = np.angle(freq_rate_2D_array[:,fringe_freq_rate_00_complex_index], deg=True)
        fringe_freq_rate_00_index         = fringe_freq_rate_00_spectrum.argmax()
        fringe_freq_rate_00_amp           = fringe_freq_rate_00_spectrum[fringe_freq_rate_00_index]
        fringe_freq_rate_00_freq          = freq_range[fringe_freq_rate_00_index]
        fringe_freq_rate_00_rate          = np.absolute(freq_rate_2D_array[fringe_freq_rate_00_index])
        fringe_freq_rate_00_phase2        = fringe_freq_rate_00_phase1[fringe_freq_rate_00_index]
        #
        # noise level, frequency domain
        #
        SNR_freq_rate, noise_level_freq = noise_level(freq_rate_2D_array, fringe_freq_rate_00_amp)
    #
    # time domain
    #
    if freq_plot != True :
        fringe_lag_rate_00_complex_index1 = np.where(rate_range==yi_time_lag )[0][0] # the direction of the lag
        fringe_lag_rate_00_complex_index2 = np.where(lag_range ==yi_time_rate)[0][0] # the direction of the rate
        fringe_lag_rate_00_lag            = np.absolute(lag_rate_2D_array[fringe_lag_rate_00_complex_index1])
        fringe_lag_rate_00_rate           = np.absolute(lag_rate_2D_array[:,fringe_lag_rate_00_complex_index2])
        fringe_lag_rate_00_amp            = np.absolute(lag_rate_2D_array[fringe_lag_rate_00_complex_index1,fringe_lag_rate_00_complex_index2])
        fringe_lag_rate_00_phase          = np.angle(lag_rate_2D_array[fringe_lag_rate_00_complex_index1,fringe_lag_rate_00_complex_index2], deg=True)
        #
        # noise level, time domain
        #
        SNR_time_lag, noise_level_lag = noise_level(lag_rate_2D_array, fringe_lag_rate_00_amp)

    #
    # fringe output
    #
    if freq_plot == True : # cross-soectrum
        if l == 0 :
            ofile_name_freq = "%s/%s_freq.txt" % (save_file_pass, save_file_name)
            output_freq  = "#************************************************************************************************************************************************************************\n"
            output_freq += "#      Epoch        Label    Source      Length     Amp       SNR      Phase     Frequency     Noise-level           %s-azel               %s-azel                       \n" % (station1_name, station2_name)
            output_freq += "#year/doy hh:mm:ss                        [s]       [%]                [deg]       [MHz]       1-sigma [%]   az[deg]  el[deg]  height[m]   az[deg]   el[deg]  height[m]  \n"
            output_freq += "#************************************************************************************************************************************************************************"
            print(output_freq); output_freq += "\n"
        output1 = "%s    %s    %s     %.2f     %f %7.1f  %+8.3f    %8.3f      %f       %.3f  %.3f  %.3f       %.3f  %.3f  %.3f" % \
            (epoch1, label, source_name, length, fringe_freq_rate_00_amp*100, SNR_freq_rate, fringe_freq_rate_00_phase2, fringe_freq_rate_00_freq, noise_level_freq*100, station1_azel[0], station1_azel[1], station1_azel[2], station2_azel[0], station2_azel[1], station2_azel[2])
        output_freq += "%s\n" % output1; print(output1)

    if freq_plot != True : # fringe
        if l == 0 :
            ofile_name_time = "%s/%s_time.txt" % (save_file_pass, save_file_name)
            output_time  = "#********************************************************************************************************************************************************\n"
            output_time += "#      Epoch        Label   Source      Length      Amp     SNR     Phase     Noise-level           %s-azel               %s-azel                      \n" % (station1_name, station2_name)
            output_time += "#year/doy hh:mm:ss                       [s]        [%]             [deg]     1-sigma[%]   az[deg]  el[deg]  height[m]   az[deg]   el[deg]  height[m]  \n"
            output_time += "#********************************************************************************************************************************************************"
            print(output_time); output_time += "\n"
        output2 = "%s    %s   %s     %.2f   %f  %7.1f  %+8.3f      %f      %.3f  %.3f  %.3f      %.3f  %.3f  %.3f" % \
            (epoch1, label, source_name, length, fringe_lag_rate_00_amp*100, SNR_time_lag, fringe_lag_rate_00_phase, noise_level_lag*100, station1_azel[0], station1_azel[1], station1_azel[2], station2_azel[0], station2_azel[1], station2_azel[2])
        output_time += "%s\n" % output2; print(output2)

        if cumulate != 0 and add_plot != True :
            cumulate_len.append(length)
            cumulate_snr.append(SNR_time_lag)
            cumulate_noise.append(noise_level_lag*100)
        
        if add_plot == True and cumulate == 0 :
            if l == 0 :
                add_plot_length = [i for i in range(length, length*loop+1, length)]
            #add_plot_length.append(length)
            add_plot_amp.append(fringe_lag_rate_00_amp*100)
            add_plot_snr.append(SNR_time_lag)
            add_plot_phase.append(fringe_lag_rate_00_phase)
            add_plot_noise_level.append(noise_level_lag*100)


    #
    # cross-spectrum
    #
    if freq_plot == True and time_plot == True :
        fig = plt.figure(figsize=(10, 7))
        gs  = GridSpec(nrows=3, ncols=2, height_ratios=[1,3,4])

        gs01 = GridSpecFromSubplotSpec(nrows=2, ncols=1, subplot_spec=gs[0:2,0], hspace=0.0, height_ratios=[1,3])
        ax1 = fig.add_subplot(gs01[0,0])
        ax2 = fig.add_subplot(gs01[1,0])
        ax1.plot(freq_range, fringe_freq_rate_00_phase1  , lw=1)
        ax2.plot(freq_range, fringe_freq_rate_00_spectrum, lw=1)
        ax2.set_xlabel("Frequency [MHz]")
        ax1.set_ylabel("Phase")
        ax2.set_ylabel("Amplitude")
        ax1.set_xlim([0,512])
        ax2.set_xlim([0,512])
        ax1.set_ylim([-180,180])
        ax2.set_ylim(ymin=0)
        ax1.xaxis.set_ticklabels([])
        ax1.yaxis.set_ticks([-90,0,90,180])
        ax1.grid(linestyle=":")
        ax2.grid(linestyle=":")

        gs3 = GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=gs[2,0])
        ax3 = fig.add_subplot(gs3[0,0])
        ax3.plot(rate_range, fringe_freq_rate_00_rate, lw=1)
        ax3.set_xlabel("Rate [Hz]")
        ax3.set_ylabel("Amplitude")
        ax3.set_xlim([rate_range[0],rate_range[-1]])
        ax3.set_ylim(ymin=0)
        ax3.grid(linestyle=":")

        gs4 = GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=gs[0:2,1])
        ax4 = fig.add_subplot(gs4[0,0])
        c = ax4.contourf(freq_range, rate_range, np.absolute(freq_rate_2D_array.T), 100, cmap="rainbow")
        fig.colorbar(c)
        ax4.set_xlabel("Frequency [MHz]")
        ax4.set_ylabel("Rate [Hz]")
        ax4.set_xlim([0,512])
        ax4.set_ylim([min(rate_range), max(rate_range)])
        ax4.grid(linestyle=":", color="black")

        gs5 = GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=gs[2,1])
        ax5 = fig.add_subplot(gs5[0,0])
        font_size = 17
        ax5.text(-0.1, 0.9, "Epoch      : %+25s"      % epoch1                         , fontsize=font_size, va="baseline", ma="right")
        ax5.text(-0.1, 0.8, "Station-1 : %+29s"       % station1_name                  , fontsize=font_size, va="baseline", ma="right")
        ax5.text(-0.1, 0.7, "Station-2 : %+29s"       % station2_name                  , fontsize=font_size, va="baseline", ma="right")
        ax5.text(-0.1, 0.6, "Source     : %+29s"      % source_name                    , fontsize=font_size)
        ax5.text(-0.1, 0.5, "Length     : %+24f [s]"  % length                         , fontsize=font_size)
        ax5.text(-0.1, 0.4, "Frequency: %+16f [MHz]"  % observing_frequency            , fontsize=font_size)
        ax5.text(-0.1, 0.3, "Peak Amp: %+23f [%%]"    % (fringe_freq_rate_00_amp * 100), fontsize=font_size)
        ax5.text(-0.1, 0.2, "Peak Phs  : %+20f [deg]" % fringe_freq_rate_00_phase2     , fontsize=font_size)
        ax5.text(-0.1, 0.1, "Peak Freq : %+17f [MHz]" % fringe_freq_rate_00_freq       , fontsize=font_size)
        ax5.text(-0.1, 0.0, "SNR      : %25f"         % SNR_freq_rate                  , fontsize=font_size)
        ax5.text(-0.1, -0.1,"1-sigma  : %+23f [%%]"   % (noise_level_freq*100)         , fontsize=font_size)
        ax5.set_axis_off()

        plt.tight_layout()
        fig.savefig("%s/%s_freq_rate_search.png" % (save_file_pass, save_file_name))
        #plt.show()
        plt.clf(); plt.close()


    #
    # delay-rate search window
    #
    if freq_plot != True and time_plot == True :

        if (-8/PP) < rate_range[0] :
            rate_min = rate_range[0]
        else :
            rate_min = (-8/PP)

        if (8/PP) < rate_range[-1] :
            rate_max = (8/PP)
        else :
            rate_max = rate_range[-1]


        fig  = plt.figure(figsize=(10, 7))
        grid = plt.GridSpec(2, 2)

        ax1 = fig.add_subplot(grid[0,0]) # delay
        ax2 = fig.add_subplot(grid[1,0]) # rate
        ax3 = fig.add_subplot(grid[0,1]) # delay-rate search window

        ax1.plot(lag_range, fringe_lag_rate_00_lag, lw=1)
        ax1.set_xlabel("Delay [Sample]")
        ax1.set_ylabel("Amplitude")
        ax1.set_xlim([-30,30])
        ax1.set_ylim(ymin=0)
        ax1.grid(linestyle=":")

        ax2.plot(rate_range, fringe_lag_rate_00_rate, lw=1)
        ax2.set_xlabel("Rate [Hz]")
        ax2.set_ylabel("Amplitude")
        ax2.set_xlim([min(rate_range), max(rate_range)])
        ax2.set_ylim(ymin=0)
        ax2.grid(linestyle=":")

        c = ax3.contourf(lag_range , rate_range , np.absolute(lag_rate_2D_array)  , 100, cmap="rainbow")
        fig.colorbar(c)
        #ax3.plot(0.0, 0.0, "x", color="white")
        ax3.set_xlabel("Delay [Sample]")
        ax3.set_ylabel("Rate [Hz]")
        ax3.set_xlim(-10,10)
        ax3.set_ylim(rate_min, rate_max)
        ax3.grid(linestyle=":", color="black")

        ax4 = fig.add_subplot(grid[1,1])
        font_size = 17
        ax4.text(-0.1, 0.9, "Epoch      : %+25s"     % epoch1                        , fontsize=font_size, va="baseline", ma="right")
        ax4.text(-0.1, 0.8, "Station-1 : %+29s"      % station1_name                 , fontsize=font_size, va="baseline", ma="right")
        ax4.text(-0.1, 0.7, "Station-2 : %+29s"      % station2_name                 , fontsize=font_size, va="baseline", ma="right")
        ax4.text(-0.1, 0.6, "Source     : %+29s"     % source_name                   , fontsize=font_size)
        ax4.text(-0.1, 0.5, "Length     : %+24f [s]" % length                        , fontsize=font_size)
        ax4.text(-0.1, 0.4, "Frequency: %+16f [MHz]" % observing_frequency           , fontsize=font_size)
        ax4.text(-0.1, 0.3, "Peak Amp: %+23f [%%]"   % (fringe_lag_rate_00_amp * 100), fontsize=font_size)
        ax4.text(-0.1, 0.2, "Peak Phs  :%20f [deg]"  % fringe_lag_rate_00_phase      , fontsize=font_size)
        ax4.text(-0.1, 0.1, "SNR      : %25f"        % SNR_time_lag                  , fontsize=font_size)
        ax4.text(-0.1, 0.0, "1-sigma : %+23f [%%]"   % (noise_level_lag*100)         , fontsize=font_size)
        #ax4.text(-0.1, -0.1, "SNR      : %25f" % SNR_time_lag, fontsize=font_size)
        ax4.set_axis_off()

        plt.tight_layout()
        fig.savefig("%s/%s_delay_rate_search.png" % (save_file_pass, save_file_name))
        #plt.show()
        plt.clf(); plt.close()

    #
    # frequency-rate 2D array & lag-rate 2D array output
    #
    if cs_output == True :
        freq_rate_2D_array_df = pd.DataFrame(freq_rate_2D_array)
        lag_rate_2D_array_df  = pd.DataFrame(lag_rate_2D_array)

        freq_rate_2D_array_df_file_name = "%s/%s_freq_rate_search.csv" % (save_file_pass, save_file_name)
        lag_rate_2D_array_df_file_name  = "%s/%s_delay_rate_search.csv" % (save_file_pass, save_file_name)

        freq_rate_2D_array_df.index   = ["%.0f" % i for i in freq_range]
        freq_rate_2D_array_df.columns = ["0" if i == 0.0 else "%.5f" % i for i in rate_range]
        freq_rate_2D_array_df.to_csv(freq_rate_2D_array_df_file_name)

        lag_rate_2D_array_df.index   = [i for i in rate_range]
        lag_rate_2D_array_df.columns = ["%.0f" % i for i in lag_range]
        lag_rate_2D_array_df.to_csv(lag_rate_2D_array_df_file_name)

    #
    # 2D-graph
    # Delay-rate Search
    #
    if DDD == True :        

        delay_rate_search_3D_lag  = np.array([lag_range]  * lag_rate_2D_array.shape[0])
        delay_rate_search_3D_rate = np.array([rate_range] * lag_rate_2D_array.shape[1]).T

        for i in range(2) :
            fig_plotly = go.Figure()
            
            if i == 0 :
                fig_plotly.add_trace(go.Surface(x=delay_rate_search_3D_lag, y=delay_rate_search_3D_rate, z=np.absolute(lag_rate_2D_array), colorscale="rainbow"))
                fig_plotly.update_layout(scene=dict(xaxis_title="Delay [sample]", xaxis = dict(range=[min(lag_range),max(lag_range)]), yaxis_title="Rate [Hz]", yaxis = dict(range=[min(rate_range),max(rate_range)]), zaxis_title="Amplitude [%]"))
            elif i == 1 : # In the enlargement graph, adjust the xrange and yrange.
                delay_rate_search_3D_lag[-15 >= delay_rate_search_3D_lag]    = np.nan
                delay_rate_search_3D_lag[+15 <= delay_rate_search_3D_lag]    = np.nan
                delay_rate_search_3D_rate[-0.1 >= delay_rate_search_3D_rate] = np.nan
                delay_rate_search_3D_rate[+0.1 <= delay_rate_search_3D_rate] = np.nan

                fig_plotly.add_trace(go.Surface(x=delay_rate_search_3D_lag, y=delay_rate_search_3D_rate, z=np.absolute(lag_rate_2D_array), colorscale="rainbow"))
                fig_plotly.update_layout(scene=dict(xaxis_title="Delay [sample]", xaxis = dict(range=[-15,15]), yaxis_title="Rate [Hz]", yaxis = dict(range=[-0.1,0.1]), zaxis_title="Amplitude [%]"))
            if i == 0 :
                fig_plotly.write_html("%s/%s_delay_rate_search_3D_GlobalImage.html" % (save_file_pass, save_file_name))
            elif i == 1 :
                fig_plotly.write_html("%s/%s_delay_rate_search_3D_Enlargement.html" % (save_file_pass, save_file_name))

#
# output
#
if output == True :
    if freq_plot != True :
        ofile_time = open(ofile_name_time, "w"); print(output_time, file=ofile_time)
    if freq_plot == True :
        ofile_freq = open(ofile_name_freq, "w"); print(output_freq, file=ofile_freq)
else :
    pass

#
# the cumulation of the integration time
#
if cumulate != 0 and add_plot != True :
    def power_law_equation(x, a, b) :
        y = a * (x)**b
        return y
    
    # length vs SNR
    param, cov = curve_fit(power_law_equation, cumulate_len, cumulate_snr)
    x_data = np.linspace(cumulate_len[0], cumulate_len[-1], 10000)
    y_data = power_law_equation(x_data, *param)

    fig = plt.figure(dpi=100, figsize=(10.24,5.12))
    plt.loglog(cumulate_len, cumulate_snr, "o", label="Source: %s\nInterval: %d s" %(source_name, cumulate))
    plt.loglog(x_data, y_data, label="Power-law fit\nIndex: %.3f" % param[1])
    plt.xlabel("Integration time [s]")
    plt.ylabel("S/N")
    plt.xlim(cumulate_len[0],cumulate_len[-1])
    plt.ylim(ymin=int(min(cumulate_snr)))
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig("%s/%s.png" % (save_file_pass, save_file_name))
    #plt.show()
    plt.clf(); plt.close()
    
    
    # Noise-level
    param, cov = curve_fit(power_law_equation, cumulate_len, cumulate_noise)
    x_data = np.linspace(cumulate_len[0], cumulate_len[-1], 10000)
    y_data = power_law_equation(x_data, *param)
    
    fig = plt.figure(dpi=100, figsize=(10.24,5.12))
    plt.loglog(cumulate_len, cumulate_noise, "o", label="Source: %s\nInterval: %d s" %(source_name, cumulate))
    plt.loglog(x_data, y_data, label="Power-law fit\nIndex: %.3f" % param[1])
    plt.xlabel("Integration time [s]")
    plt.ylabel("Noise-level [%]")
    plt.xlim(cumulate_len[0],cumulate_len[-1])
    plt.ylim(ymin=int(min(cumulate_noise)))
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig("%s/%s_noise-level.png" % (save_file_pass, save_file_name))
    #plt.show()
    plt.clf(); plt.close()


#
# plot the amplitude, the phase, the S/N and the noise level in the time domain
#
if add_plot == True and cumulate == 0 :
    time_add_plot(add_plot_length, add_plot_amp        , epoch1, "Amplitude [%]" , "amp"   , add_plot_length[0], add_plot_length[-1])
    time_add_plot(add_plot_length, add_plot_snr        , epoch1, "S/N"            , "snr"  , add_plot_length[0], add_plot_length[-1])
    time_add_plot(add_plot_length, add_plot_phase      , epoch1, "Phase [deg]"    , "phase", add_plot_length[0], add_plot_length[-1])
    time_add_plot(add_plot_length, add_plot_noise_level, epoch1, "Noise Level [%]", "noise", add_plot_length[0], add_plot_length[-1])


#
# dynamic spectrum: the frequency domain & the time domain
#
if ds_plot != False :
    dynamic_spectrum_freq_time = []
    dynamic_spectrum_lag_time = []

    for i in range(fft_point//2) :
        dynamic_spectrum_freq_time.append(integ_complex_2D_array[:,i]) # lag vs integ time in the dynamic spectrum

    for i in range(PP) :
        ifft_time_direction = np.fft.ifft(integ_complex_2D_array[i], n=fft_point) # convert the frequency domain to the time domain by executing the IFFT in the frequency
        dynamic_spectrum_lag_time.append(ifft_time_direction)

    dynamic_spectrum_freq_time = np.array(dynamic_spectrum_freq_time)
    dynamic_spectrum_lag_time  = np.array(dynamic_spectrum_lag_time)
    dynamic_spectrum_lag_time  = np.concatenate([dynamic_spectrum_lag_time[:, fft_point//2:], dynamic_spectrum_lag_time[:, :fft_point//2]], 1)
    dynamic_spectrum_lag_time  = dynamic_spectrum_lag_time[:, ::-1]

    dynamic_spectrum(freq_range, integ_range, dynamic_spectrum_freq_time.T, PP, 1) # the frequency domain: length vs frequency and cross-spectrum
    dynamic_spectrum(lag_range , integ_range, dynamic_spectrum_lag_time   , PP, 2) # the time domain: length vs the frequency domain


