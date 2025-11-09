"""手写数字识别器 - 快速响应版"""

import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageDraw
from neural_network import TwoLayerNN
import time
import os


class FastDigitRecognizerGUI:
    """手写数字识别GUI，响应<1ms"""
    
    def __init__(self, model):
        self.model = model
        
        self.root = tk.Tk()
        self.root.title("手写数字识别器 [快速版]")
        self.root.geometry("600x520")
        self.root.resizable(False, False)
        
        # 画布设置
        self.canvas_size = 280
        self.image_size = 28
        
        # 创建PIL图像用于预处理
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), "white")
        self.draw = ImageDraw.Draw(self.image)
        
        # 绘制状态
        self.is_drawing = False
        self.last_x = None
        self.last_y = None
        
        # 统计信息
        self.recognition_count = 0
        self.total_time = 0
        
        self.create_widgets()
        
    def create_widgets(self):
        """创建界面组件"""
        
        # 标题
        title_frame = tk.Frame(self.root, bg="#2196F3", pady=10)
        title_frame.pack(fill=tk.X)
        
        title_label = tk.Label(
            title_frame,
            text="⚡ 手写数字识别器 - 快速版",
            font=("Arial", 18, "bold"),
            bg="#2196F3",
            fg="white"
        )
        title_label.pack()
        
        # 说明
        instruction_label = tk.Label(
            self.root,
            text="请在下方画布绘制数字 (0-9)，点击识别或按回车",
            font=("Arial", 11),
            pady=8
        )
        instruction_label.pack()
        
        # 画布框架
        canvas_frame = tk.Frame(self.root, relief=tk.RAISED, borderwidth=2, bg="#E3F2FD")
        canvas_frame.pack(pady=10)
        
        # 画布
        self.canvas = tk.Canvas(
            canvas_frame,
            width=self.canvas_size,
            height=self.canvas_size,
            bg="white",
            cursor="crosshair"
        )
        self.canvas.pack(padx=5, pady=5)
        
        # 绑定事件
        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw_line)
        self.canvas.bind("<ButtonRelease-1>", self.stop_draw)
        
        # 绑定键盘回车键
        self.root.bind("<Return>", lambda e: self.recognize_digit())
        
        # 结果框架
        result_frame = tk.Frame(self.root, bg="#E8F5E9", relief=tk.RIDGE, borderwidth=2)
        result_frame.pack(pady=10, padx=20, fill=tk.X)
        
        result_inner = tk.Frame(result_frame, bg="#E8F5E9", pady=10)
        result_inner.pack()
        
        # 识别结果
        tk.Label(
            result_inner,
            text="识别结果:",
            font=("Arial", 13),
            bg="#E8F5E9"
        ).grid(row=0, column=0, padx=5)
        
        self.prediction_label = tk.Label(
            result_inner,
            text="--",
            font=("Arial", 32, "bold"),
            fg="#1976D2",
            bg="#E8F5E9",
            width=3
        )
        self.prediction_label.grid(row=0, column=1, padx=10)
        
        # 置信度
        tk.Label(
            result_inner,
            text="置信度:",
            font=("Arial", 13),
            bg="#E8F5E9"
        ).grid(row=0, column=2, padx=5)
        
        self.confidence_label = tk.Label(
            result_inner,
            text="--",
            font=("Arial", 16, "bold"),
            fg="#388E3C",
            bg="#E8F5E9"
        )
        self.confidence_label.grid(row=0, column=3, padx=5)
        
        # 识别时间
        tk.Label(
            result_inner,
            text="用时:",
            font=("Arial", 11),
            bg="#E8F5E9"
        ).grid(row=0, column=4, padx=5)
        
        self.time_label = tk.Label(
            result_inner,
            text="--",
            font=("Arial", 11),
            fg="#F57C00",
            bg="#E8F5E9"
        )
        self.time_label.grid(row=0, column=5, padx=5)
        
        # 概率分布（简化显示）
        prob_frame = tk.Frame(self.root)
        prob_frame.pack(pady=5)
        
        tk.Label(
            prob_frame,
            text="各数字概率分布:",
            font=("Arial", 10, "bold")
        ).pack()
        
        self.prob_bars = []
        bars_container = tk.Frame(prob_frame)
        bars_container.pack(pady=5)
        
        for i in range(10):
            digit_frame = tk.Frame(bars_container)
            digit_frame.grid(row=0, column=i, padx=1)
            
            tk.Label(digit_frame, text=str(i), font=("Arial", 9)).pack()
            
            bar = ttk.Progressbar(
                digit_frame,
                orient=tk.VERTICAL,
                length=60,
                mode='determinate'
            )
            bar.pack()
            self.prob_bars.append(bar)
        
        # 按钮框架
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=10)
        
        # 识别按钮
        recognize_button = tk.Button(
            button_frame,
            text="🔍 识别 (Enter)",
            font=("Arial", 11, "bold"),
            bg="#4CAF50",
            fg="white",
            width=12,
            height=1,
            command=self.recognize_digit,
            cursor="hand2"
        )
        recognize_button.grid(row=0, column=0, padx=5)
        
        # 清除按钮
        clear_button = tk.Button(
            button_frame,
            text="🗑️ 清除",
            font=("Arial", 11),
            bg="#FF9800",
            fg="white",
            width=12,
            height=1,
            command=self.clear_canvas,
            cursor="hand2"
        )
        clear_button.grid(row=0, column=1, padx=5)
        
        # 退出按钮
        quit_button = tk.Button(
            button_frame,
            text="❌ 退出",
            font=("Arial", 11),
            bg="#607D8B",
            fg="white",
            width=12,
            height=1,
            command=self.root.quit,
            cursor="hand2"
        )
        quit_button.grid(row=0, column=2, padx=5)
        
        # 状态栏
        self.status_label = tk.Label(
            self.root,
            text="准备就绪 | 平均响应时间: -- ms",
            font=("Arial", 9),
            bg="#ECEFF1",
            fg="#37474F",
            anchor=tk.W,
            padx=10,
            pady=3
        )
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)
        
    def start_draw(self, event):
        """开始绘制"""
        self.is_drawing = True
        self.last_x = event.x
        self.last_y = event.y
        
    def draw_line(self, event):
        """绘制线条"""
        if self.is_drawing:
            x, y = event.x, event.y
            
            # 在画布上绘制
            self.canvas.create_line(
                self.last_x, self.last_y, x, y,
                width=18,
                fill="black",
                capstyle=tk.ROUND,
                smooth=True
            )
            
            # 在 PIL 图像上绘制
            self.draw.line(
                [self.last_x, self.last_y, x, y],
                fill="black",
                width=18
            )
            
            self.last_x = x
            self.last_y = y
    
    def stop_draw(self, event):
        """停止绘制"""
        self.is_drawing = False
        
    def clear_canvas(self):
        """清除画布"""
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), "white")
        self.draw = ImageDraw.Draw(self.image)
        
        self.prediction_label.config(text="--")
        self.confidence_label.config(text="--")
        self.time_label.config(text="--")
        for bar in self.prob_bars:
            bar['value'] = 0
            
    def preprocess_image(self):
        """预处理手写输入
        
        关键优化:
        1. 形态学膨胀加粗线条 (解决0→4/9误识别)
        2. 边界框裁剪+居中
        3. 缩放到28x28
        """
        from scipy.ndimage import binary_dilation
        
        img_array = np.array(self.image, dtype=np.float32)
        img_array = 255.0 - img_array  # 反转: 白底黑字 -> 黑底白字
        
        # 二值化
        binary_img = (img_array > 30).astype(np.uint8)
        
        # 膨胀加粗 (这步很关键!)
        structure = np.ones((3, 3), dtype=np.uint8)
        dilated = binary_dilation(binary_img, structure=structure, iterations=2)
        img_array = dilated.astype(np.float32) * 255.0
        
        # 裁剪到内容区域
        rows = np.any(img_array > 0, axis=1)
        cols = np.any(img_array > 0, axis=0)
        
        if rows.sum() > 0 and cols.sum() > 0:
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            cropped = img_array[rmin:rmax+1, cmin:cmax+1]
            
            # 加边距
            h, w = cropped.shape
            margin = max(int(h * 0.2), int(w * 0.2))
            new_size = max(h, w) + margin * 2
            padded = np.zeros((new_size, new_size), dtype=np.float32)
            
            y_offset = (new_size - h) // 2
            x_offset = (new_size - w) // 2
            padded[y_offset:y_offset+h, x_offset:x_offset+w] = cropped
            
            # 缩放到28x28
            from PIL import Image as PILImage
            padded_img = PILImage.fromarray(padded.astype(np.uint8))
            img_resized = padded_img.resize((28, 28), PILImage.BILINEAR)
            img_2d = np.array(img_resized, dtype=np.float32)
        else:
            img_2d = np.zeros((28, 28), dtype=np.float32)
        
        # 归一化
        img_2d = img_2d / 255.0
        return img_2d.reshape(1, -1)
    
    def recognize_digit(self):
        """识别手写数字，<1ms响应"""
        start_time = time.time()
        
        try:
            img_data = self.preprocess_image()
            
            # 检查空白
            if img_data.max() < 0.1:
                self.prediction_label.config(text="?", fg="red")
                self.confidence_label.config(text="空白", fg="red")
                self.time_label.config(text="--")
                return
            
            # 前向传播
            probabilities = self.model.forward(img_data)[0]
            prediction = np.argmax(probabilities)
            confidence = probabilities[prediction]
            
            elapsed_ms = (time.time() - start_time) * 1000
            
            # 更新统计
            self.recognition_count += 1
            self.total_time += elapsed_ms
            avg_time = self.total_time / self.recognition_count
            
            # 显示结果
            self.prediction_label.config(text=str(prediction), fg="#1976D2")
            self.confidence_label.config(
                text=f"{confidence*100:.1f}%",
                fg="#388E3C" if confidence > 0.7 else "#F57C00"
            )
            self.time_label.config(text=f"{elapsed_ms:.1f}ms")
            
            # 更新概率条
            for i, prob in enumerate(probabilities):
                self.prob_bars[i]['value'] = prob * 100
            
            # 更新状态栏
            self.status_label.config(
                text=f"✓ 识别完成 | 平均响应: {avg_time:.1f}ms | 总次数: {self.recognition_count}"
            )
            
            print(f"识别: {prediction} | 置信度: {confidence*100:.1f}% | 用时: {elapsed_ms:.1f}ms")
            
        except Exception as e:
            print(f"识别错误: {e}")
            self.prediction_label.config(text="错误", fg="red")
            
    def run(self):
        """运行 GUI"""
        print("\n" + "=" * 50)
        print("⚡ 手写数字识别器 [快速版] 已启动")
        print("=" * 50)
        print("\n使用提示:")
        print("  • 在画布上绘制数字 (0-9)")
        print("  • 点击 [识别] 或按 [Enter] 键")
        print("  • 点击 [清除] 重新绘制")
        print("  • 响应时间 < 100ms")
        print("=" * 50 + "\n")
        
        self.root.mainloop()


def load_model_fast(model_path='mnist_digit_recognizer.npz'):
    """快速加载模型"""
    print("⚡ 快速加载模式")
    print("-" * 50)
    
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        print("\n正在启动快速训练...")
        
        # 自动运行快速训练
        try:
            import quick_train
            model = quick_train.quick_train_model()
            if model is None:
                return None
        except Exception as e:
            print(f"训练失败: {e}")
            return None
    
    print(f"📂 加载模型: {model_path}")
    start_time = time.time()
    
    # 创建模型（与训练时保持一致）
    model = TwoLayerNN(
        input_size=784,
        hidden_size=300,  # 与训练脚本一致（超高准确率）
        output_size=10
    )
    
    try:
        # 加载权重
        model.load_weights(model_path)
        
        # 预热
        dummy = np.random.randn(1, 784) / 255.0
        _ = model.forward(dummy)
        
        elapsed_ms = (time.time() - start_time) * 1000
        print(f"✓ 模型加载成功！耗时: {elapsed_ms:.1f} ms")
        print("-" * 50 + "\n")
        
        return model
        
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return None


if __name__ == "__main__":
    # 快速加载模型
    model = load_model_fast()
    
    if model is not None:
        # 启动 GUI
        app = FastDigitRecognizerGUI(model)
        app.run()
    else:
        print("\n无法启动识别器，请检查模型文件")
