import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
from keras.models import load_model
# import matplotlib.pyplot as plt

# モデルの読み込み
model = load_model('C:/Briefcase/python/cnn_model_mnist.h5') # パスは適宜変更

# 描画領域のサイズ
width = 280
height = 280

# 描画領域をリセットする（画像データ上は黒(0)で埋める）関数
def reset_canvas():
    canvas.delete("all")
    draw.rectangle((0, 0, width, height), fill=(0)) 

# 画像を保存して予測する関数
def predict_digit():
    image.save('digit.png')
    # 画像を読み込む際に28x28にリサイズ
    image_resized = Image.open('digit.png').resize((28,28)).convert('L')  
    image_arr = np.array(image_resized)

    # 画像の表示(確認用)
    # plt.imshow(image_arr, cmap='gray')
    # plt.show()

    image_arr = image_arr.reshape(1, 28, 28, 1)
    image_arr = image_arr.astype('float32')
    image_arr /= 255
    prediction = model.predict([image_arr])
    predicted_label = np.argmax(prediction, axis=1)
    print("Predicted Label: ", predicted_label[0])

# ウィンドウとキャンバス（白地）の作成
window = tk.Tk()
canvas = tk.Canvas(window, width=width, height=height, bg='white')
canvas.pack()

# 描画領域のリセットボタン
reset_button = tk.Button(window, text='Reset', command=reset_canvas)
reset_button.pack()

# 予測ボタン
predict_button = tk.Button(window, text='Predict', command=predict_digit)
predict_button.pack()

# 描画用のイメージ（画像データの初期状態は黒（0））
image = Image.new('L', (width, height), (0))
draw = ImageDraw.Draw(image)

# マウスのドラッグで描画（太さはx or y + ** の数字で決まる；12くらいがMNISTデータと近そう）
def draw_lines(event):
    x, y = event.x, event.y
    canvas.create_oval(x, y, x + 12, y + 12, fill='black', width=0)  # GUI上では黒（0）で描画
    draw.ellipse([x, y, x + 12, y + 12], fill='white')  # 画像上では白（255）で描画

canvas.bind("<B1-Motion>", draw_lines)

window.mainloop()