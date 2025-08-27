import sys
import cv2
import time
import serial
import threading
import numpy as np
from PyQt6 import QtWidgets, uic, QtGui
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QIcon, QColor, QLinearGradient, QPainter, QPen, QFont, QAction
from ultralytics import YOLO
import serial.tools.list_ports
import random
import os
from pathlib import Path
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import mysql.connector
from datetime import datetime
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import queue
import csv
import websocket
import json

# Cấu hình ghi log
logging.basicConfig(filename='motor_control.log', level=logging.DEBUG, 
                   format='%(asctime)s - %(levelname)s - %(message)s')

# Cấu hình
BASE_DIR = Path(__file__).parent
UI_FILE = BASE_DIR / "motor_1.ui"
MODEL_PATH = BASE_DIR / "banyoloN.pt"
CONTROL_FILE = BASE_DIR / "control.txt"
DQN_MODEL_FILE = BASE_DIR / "dqn_policy_net.pth"
CSV_FILE = BASE_DIR / "sensor_data_export.csv"
OVERLOAD_THRESHOLD = 1000
TEMP_THRESHOLD = 60
MAX_POINTS = 100
HISTORY_HOURS = 24
SERIAL_TIMEOUT = 2  # Giây
WEBSOCKET_URL = "ws://10.217.159.15:3001"

# Siêu tham số DQN
ACTIONS = [0, 25, 50, 75, 100, 125, 150, 175, 200]
EPSILON = 0.2
MIN_EPSILON = 0.01
EPSILON_DECAY = 0.99
GAMMA = 0.95
REPLAY_BUFFER_SIZE = 10000
BATCH_SIZE = 32
LEARNING_RATE = 0.001
SPEED_CHANGE_LIMIT = 25
SMOOTHNESS_WINDOW = 5
TARGET_UPDATE_FREQ = 100

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

class GaugeWidget(QtWidgets.QWidget):
    def __init__(self, title="", min_val=0, max_val=100, unit=""):
        super().__init__()
        self.title = title
        self.min_val = min_val
        self.max_val = max_val
        self.unit = unit
        self.current_value = min_val
        self.setMinimumSize(200, 200)
        self.init_ui()

    def init_ui(self):
        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)
        self.title_label = QtWidgets.QLabel(self.title)
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(self.title_label)
        self.value_label = QtWidgets.QLabel(f"{self.current_value} {self.unit}")
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.value_label.setStyleSheet("font-size: 24px;")
        layout.addWidget(self.value_label)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        size = min(self.width(), self.height()) - 20
        x = (self.width() - size) / 2
        y = (self.height() - size) / 2 + 20
        painter.setPen(QPen(QColor(200, 200, 200), 2))
        painter.drawArc(int(x), int(y), int(size), int(size), 0, 180 * 16)
        gradient = QLinearGradient(0, 0, 0, self.height())
        gradient.setColorAt(0, QColor(0, 200, 0))
        gradient.setColorAt(0.5, QColor(200, 200, 0))
        gradient.setColorAt(1, QColor(200, 0, 0))
        pen = QPen(gradient, 10)
        painter.setPen(pen)
        angle = 180 * (self.current_value - self.min_val) / (self.max_val - self.min_val)
        painter.drawArc(int(x+5), int(y+5), int(size-10), int(size-10), 180 * 16, -int(angle * 16))
        painter.setPen(QPen(Qt.GlobalColor.black, 2))
        font = QFont()
        font.setPointSize(8)
        painter.setFont(font)
        for i in range(0, 181, 30):
            angle = i - 180
            rad = angle * np.pi / 180
            inner_x = x + size/2 + (size/2 - 15) * np.cos(rad)
            inner_y = y + size/2 + (size/2 - 15) * np.sin(rad)
            outer_x = x + size/2 + (size/2 - 5) * np.cos(rad)
            outer_y = y + size/2 + (size/2 - 5) * np.sin(rad)
            painter.drawLine(int(inner_x), int(inner_y), int(outer_x), int(outer_y))
            if i % 60 == 0:
                value = self.min_val + (self.max_val - self.min_val) * i / 180
                text_x = x + size/2 + (size/2 - 25) * np.cos(rad)
                text_y = y + size/2 + (size/2 - 25) * np.sin(rad)
                painter.drawText(int(text_x - 10), int(text_y - 10), 20, 20, 
                               Qt.AlignmentFlag.AlignCenter, f"{value:.0f}")

    def update_value(self, value):
        self.current_value = value
        self.value_label.setText(f"{self.current_value:.1f} {self.unit}")
        self.update()

class SpeedGauge(GaugeWidget):
    def __init__(self):
        super().__init__(title="Tốc độ động cơ", min_val=0, max_val=200, unit="RPM")
        self.setStyleSheet("background-color: rgba(255, 255, 255, 150); border-radius: 10px;")

class TemperatureGauge(GaugeWidget):
    def __init__(self):
        super().__init__(title="Nhiệt độ", min_val=0, max_val=100, unit="°C")
        self.setStyleSheet("background-color: rgba(255, 255, 255, 150); border-radius: 10px;")

class WeightGauge(GaugeWidget):
    def __init__(self):
        super().__init__(title="Trọng lượng", min_val=0, max_val=1000, unit="g")
        self.setStyleSheet("background-color: rgba(255, 255, 255, 150); border-radius: 10px;")

class SetupDialog(QtWidgets.QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cài đặt kết nối")
        self.setFixedSize(400, 300)
        layout = QtWidgets.QVBoxLayout()
        self.tab_widget = QtWidgets.QTabWidget()
        connection_tab = QtWidgets.QWidget()
        connection_layout = QtWidgets.QVBoxLayout()
        self.com_label = QtWidgets.QLabel("Chọn cổng COM:")
        self.com_combo = QtWidgets.QComboBox()
        ports = serial.tools.list_ports.comports()
        if not ports:
            self.com_combo.addItem("Không tìm thấy cổng COM")
        else:
            for port in ports:
                self.com_combo.addItem(port.device)
        self.camera_label = QtWidgets.QLabel("Nhập cổng camera (0, 1, hoặc URL):")
        self.camera_input = QtWidgets.QLineEdit("0")
        connection_layout.addWidget(self.com_label)
        connection_layout.addWidget(self.com_combo)
        connection_layout.addWidget(self.camera_label)
        connection_layout.addWidget(self.camera_input)
        connection_tab.setLayout(connection_layout)
        control_tab = QtWidgets.QWidget()
        control_layout = QtWidgets.QVBoxLayout()
        self.mode_label = QtWidgets.QLabel("Chọn chế độ điều khiển:")
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["Normal", "Fuzzy", "RL"])
        self.threshold_label = QtWidgets.QLabel("Ngưỡng quá tải (g):")
        self.threshold_input = QtWidgets.QLineEdit(str(OVERLOAD_THRESHOLD))
        self.temp_label = QtWidgets.QLabel("Ngưỡng nhiệt độ (°C):")
        self.temp_input = QtWidgets.QLineEdit(str(TEMP_THRESHOLD))
        control_layout.addWidget(self.mode_label)
        control_layout.addWidget(self.mode_combo)
        control_layout.addWidget(self.threshold_label)
        control_layout.addWidget(self.threshold_input)
        control_layout.addWidget(self.temp_label)
        control_layout.addWidget(self.temp_input)
        control_tab.setLayout(control_layout)
        self.tab_widget.addTab(connection_tab, "Kết nối")
        self.tab_widget.addTab(control_tab, "Điều khiển")
        layout.addWidget(self.tab_widget)
        button_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok | 
                                               QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        self.setLayout(layout)

    def get_settings(self):
        try:
            camera_port = self.camera_input.text()
            if not (camera_port.isdigit() or camera_port.startswith('http')):
                raise ValueError("Cổng camera phải là số hoặc URL hợp lệ!")
            return {
                'com_port': self.com_combo.currentText(),
                'camera_port': camera_port,
                'control_mode': self.mode_combo.currentText(),
                'overload_threshold': float(self.threshold_input.text()),
                'temp_threshold': float(self.temp_input.text())
            }
        except ValueError as e:
            QtWidgets.QMessageBox.critical(self, "Lỗi", str(e))
            return None

class MotorControlApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        if not UI_FILE.exists():
            QtWidgets.QMessageBox.critical(self, "Lỗi", "Không tìm thấy file UI!")
            sys.exit(1)

        # Hiển thị cửa sổ cài đặt
        self.setup_dialog = SetupDialog()
        if not self.setup_dialog.exec() or not self.setup_dialog.get_settings():
            QtWidgets.QMessageBox.critical(self, "Lỗi", "Cần hoàn tất cài đặt để tiếp tục!")
            sys.exit(0)

        settings = self.setup_dialog.get_settings()
        self.serial_port = settings['com_port']
        self.camera_port = settings['camera_port']
        self.control_mode = settings['control_mode']
        self.overload_threshold = settings['overload_threshold']
        self.temp_threshold = settings['temp_threshold']

        uic.loadUi(UI_FILE, self)
        self.setWindowTitle("Hệ thống Điều khiển Động cơ Công nghiệp")
        self.setWindowIcon(QIcon('motor_icon.png'))

        # Khởi tạo các thành phần giao diện
        self.video_label = self.findChild(QtWidgets.QLabel, "videoLabel")
        self.count_label = self.findChild(QtWidgets.QLabel, "countLabel")
        self.weight_label = self.findChild(QtWidgets.QLabel, "weightLabel")
        self.speed_label = self.findChild(QtWidgets.QLabel, "speedLabel")
        self.alert_label = self.findChild(QtWidgets.QLabel, "alertLabel")
        self.temp_label = self.findChild(QtWidgets.QLabel, "tempLabel")
        self.detail_label = self.findChild(QtWidgets.QLabel, "detailLabel")
        self.toggle_button = self.findChild(QtWidgets.QPushButton, "toggleButton")
        self.motor_status_label = self.findChild(QtWidgets.QLabel, "motorStatusLabel")
        self.motor_dot = self.findChild(QtWidgets.QLabel, "motorStatusDot")
        self.alert_dot = self.findChild(QtWidgets.QLabel, "alertStatusDot")
        self.control_tabs = self.findChild(QtWidgets.QTabWidget, "controlTabs")
        self.export_button = self.findChild(QtWidgets.QPushButton, "exportButton")

        # Thêm gauges
        self.gauges_layout = QtWidgets.QHBoxLayout()
        self.speed_gauge = SpeedGauge()
        self.temp_gauge = TemperatureGauge()
        self.weight_gauge = WeightGauge()
        self.gauges_layout.addWidget(self.speed_gauge)
        self.gauges_layout.addWidget(self.temp_gauge)
        self.gauges_layout.addWidget(self.weight_gauge)
        self.findChild(QtWidgets.QVBoxLayout, "verticalLayout_2").insertLayout(1, self.gauges_layout)

        # Kết nối sự kiện
        self.export_button.clicked.connect(self.export_to_csv)
        self.actionExit = self.findChild(QAction, "actionExit")
        self.actionFullscreen = self.findChild(QAction, "actionFullscreen")
        self.actionAbout = self.findChild(QAction, "actionAbout")
        if self.actionExit:
            self.actionExit.triggered.connect(self.close)
        if self.actionFullscreen:
            self.actionFullscreen.triggered.connect(self.toggle_fullscreen)
        if self.actionAbout:
            self.actionAbout.triggered.connect(self.show_about)

        # Khởi tạo biến
        self.motor_enabled = False
        self.total_count = 0
        self.weight = 0
        self.temperature = 0
        self.last_sent_speed = 0
        self.last_speed = 100  # Khởi tạo tốc độ trước đó
        self.serial_connected = False
        self.epsilon = EPSILON
        self.step_count = 0
        self.start_time = time.time()
        self.normal_times = []
        self.normal_weights = []
        self.normal_counts = []
        self.normal_speeds = []
        self.fuzzy_times = []
        self.fuzzy_weights = []
        self.fuzzy_counts = []
        self.fuzzy_speeds = []
        self.rl_times = []
        self.rl_rewards = []
        self.rl_speeds = []
        self.rl_weights = []
        self.rl_counts = []
        self.rl_epsilons = []
        self.history_data = []
        self.frame_count = 0
        self.last_results = None
        self.serial_queue = queue.Queue()
        self.pending_commands = {}

        # Khởi tạo DQN
        self.state_size = 4  # [weight, count, last_speed, avg_speed]
        self.action_size = len(ACTIONS)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(self.state_size, self.action_size).to(self.device)
        self.target_net = DQN(self.state_size, self.action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)
        if DQN_MODEL_FILE.exists():
            self.policy_net.load_state_dict(torch.load(DQN_MODEL_FILE, map_location=self.device))
            self.target_net.load_state_dict(self.policy_net.state_dict())
            logging.info("Loaded DQN policy network from file")

        # Khởi tạo WebSocket
        self.ws_client = None
        self.init_websocket()

        # Tải mô hình YOLO
        if not MODEL_PATH.exists():
            QtWidgets.QMessageBox.critical(self, "Lỗi", "Không tìm thấy file mô hình YOLO!")
            sys.exit(1)
        self.model = YOLO(MODEL_PATH, task='detect')
        self.model.to('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"YOLO model loaded on {'CUDA' if torch.cuda.is_available() else 'CPU'}")
        self.label_names = self.model.names
        self.class_labels = ['branch', 'coal', 'iron', 'stick', 'stone']
        self.class_counts = {label: 0 for label in self.class_labels}

        # ROI và đường line
        self.roi_polygon = np.array([[481, 3], [550, 479], [145, 477], [233, 2], [233, 2]], dtype=np.int32)
        self.line_start = (180, 276)
        self.line_end = (519, 279)
        self.line_y = self.line_start[1]
        self.prev_centers = []

        self.init_fuzzy_system()
        self.init_database()
        self.init_serial()
        self.init_camera()

        if self.serial_connected:
            self.serial_thread = threading.Thread(target=self.read_serial_data)
            self.serial_thread.daemon = True
            self.serial_thread.start()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(200)

        self.control_timer = QTimer()
        self.control_timer.timeout.connect(self.read_control_file)
        self.control_timer.start(200)

        self.history_timer = QTimer()
        self.history_timer.timeout.connect(self.update_history)
        self.history_timer.start(60000)

        self.toggle_button.clicked.connect(self.toggle_motor)
        self.update_dot(self.motor_dot, "#000000")
        self.motor_status_label.setText("Động cơ: Tắt")
        self.init_plots()
        self.load_history_data()

    def init_fuzzy_system(self):
        weight = ctrl.Antecedent(np.arange(0, 1001, 1), 'weight')
        count = ctrl.Antecedent(np.arange(0, 11, 0.1), 'count')
        speed = ctrl.Consequent(np.arange(0, 201, 1), 'speed')
        weight['extremely_low'] = fuzz.trimf(weight.universe, [0, 0, 50])
        weight['very_low'] = fuzz.trimf(weight.universe, [0, 50, 100])
        weight['low'] = fuzz.trimf(weight.universe, [50, 100, 200])
        weight['medium_low'] = fuzz.trimf(weight.universe, [100, 200, 400])
        weight['medium'] = fuzz.trimf(weight.universe, [200, 400, 600])
        weight['high'] = fuzz.trimf(weight.universe, [400, 600, 800])
        weight['very_high'] = fuzz.trimf(weight.universe, [600, 800, 1000])
        count['zero'] = fuzz.trimf(count.universe, [0, 0, 1])
        count['very_few'] = fuzz.trimf(count.universe, [0, 1, 2])
        count['few'] = fuzz.trimf(count.universe, [1, 2, 4])
        count['medium'] = fuzz.trimf(count.universe, [2, 4, 6])
        count['many'] = fuzz.trimf(count.universe, [4, 6, 8])
        count['very_many'] = fuzz.trimf(count.universe, [6, 8, 10])
        count['extremely_many'] = fuzz.trimf(count.universe, [8, 10, 10])
        speed['stop'] = fuzz.trimf(speed.universe, [0, 0, 75])
        speed['very_low'] = fuzz.trimf(speed.universe, [75, 90, 100])
        speed['low'] = fuzz.trimf(speed.universe, [100, 125, 135])
        speed['medium_low'] = fuzz.trimf(speed.universe, [125, 140, 150])
        speed['medium'] = fuzz.trimf(speed.universe, [140, 150, 160])
        speed['medium_high'] = fuzz.trimf(speed.universe, [150, 165, 175])
        speed['high'] = fuzz.trimf(speed.universe, [165, 175, 200])
        speed['very_high'] = fuzz.trimf(speed.universe, [175, 200, 200])
        rules = [
            ctrl.Rule(count['zero'] & weight['extremely_low'], speed['very_high']),
            ctrl.Rule(count['zero'] & weight['very_low'], speed['medium_high']),
            ctrl.Rule(count['zero'] & weight['low'], speed['medium']),
            ctrl.Rule(count['zero'] & weight['medium_low'], speed['medium_low']),
            ctrl.Rule(count['zero'] & weight['medium'], speed['medium_low']),
            ctrl.Rule(count['zero'] & weight['high'], speed['medium_low']),
            ctrl.Rule(count['zero'] & weight['very_high'], speed['low']),
            ctrl.Rule(count['very_few'] & weight['extremely_low'], speed['medium_low']),
            ctrl.Rule(count['very_few'] & weight['very_low'], speed['medium_low']),
            ctrl.Rule(count['very_few'] & weight['low'], speed['medium_low']),
            ctrl.Rule(count['very_few'] & weight['medium_low'], speed['low']),
            ctrl.Rule(count['very_few'] & weight['medium'], speed['low']),
            ctrl.Rule(count['very_few'] & weight['high'], speed['very_low']),
            ctrl.Rule(count['very_few'] & weight['very_high'], speed['very_low']),
            ctrl.Rule(count['few'] & weight['extremely_low'], speed['low']),
            ctrl.Rule(count['few'] & weight['very_low'], speed['low']),
            ctrl.Rule(count['few'] & weight['low'], speed['low']),
            ctrl.Rule(count['few'] & weight['medium_low'], speed['very_low']),
            ctrl.Rule(count['few'] & weight['medium'], speed['very_low']),
            ctrl.Rule(count['few'] & weight['high'], speed['stop']),
            ctrl.Rule(count['few'] & weight['very_high'], speed['stop']),
            ctrl.Rule(count['medium'] & weight['extremely_low'], speed['very_low']),
            ctrl.Rule(count['medium'] & weight['very_low'], speed['very_low']),
            ctrl.Rule(count['medium'] & weight['low'], speed['very_low']),
            ctrl.Rule(count['medium'] & weight['medium_low'], speed['stop']),
            ctrl.Rule(count['medium'] & weight['medium'], speed['stop']),
            ctrl.Rule(count['medium'] & weight['high'], speed['stop']),
            ctrl.Rule(count['medium'] & weight['very_high'], speed['stop']),
            ctrl.Rule(count['many'] & weight['extremely_low'], speed['stop']),
            ctrl.Rule(count['many'] & weight['very_low'], speed['stop']),
            ctrl.Rule(count['many'] & weight['low'], speed['stop']),
            ctrl.Rule(count['many'] & weight['medium_low'], speed['stop']),
            ctrl.Rule(count['many'] & weight['medium'], speed['stop']),
            ctrl.Rule(count['many'] & weight['high'], speed['stop']),
            ctrl.Rule(count['many'] & weight['very_high'], speed['stop']),
            ctrl.Rule(count['very_many'] & weight['extremely_low'], speed['stop']),
            ctrl.Rule(count['very_many'] & weight['very_low'], speed['stop']),
            ctrl.Rule(count['very_many'] & weight['low'], speed['stop']),
            ctrl.Rule(count['very_many'] & weight['medium_low'], speed['stop']),
            ctrl.Rule(count['very_many'] & weight['medium'], speed['stop']),
            ctrl.Rule(count['very_many'] & weight['high'], speed['stop']),
            ctrl.Rule(count['very_many'] & weight['very_high'], speed['stop']),
            ctrl.Rule(count['extremely_many'] & weight['extremely_low'], speed['stop']),
            ctrl.Rule(count['extremely_many'] & weight['very_low'], speed['stop']),
            ctrl.Rule(count['extremely_many'] & weight['low'], speed['stop']),
            ctrl.Rule(count['extremely_many'] & weight['medium_low'], speed['stop']),
            ctrl.Rule(count['extremely_many'] & weight['medium'], speed['stop']),
            ctrl.Rule(count['extremely_many'] & weight['high'], speed['stop']),
            ctrl.Rule(count['extremely_many'] & weight['very_high'], speed['stop']),
        ]
        fuzzy_ctrl = ctrl.ControlSystem(rules)
        self.fuzzy_sim = ctrl.ControlSystemSimulation(fuzzy_ctrl)
        logging.info("Fuzzy system initialized with strong count priority and max speed 200 RPM")

    def get_state(self, weight, count, last_speed):
        recent_speeds = self.rl_speeds[-SMOOTHNESS_WINDOW:] if len(self.rl_speeds) >= SMOOTHNESS_WINDOW else self.rl_speeds
        avg_speed = np.mean(recent_speeds) if recent_speeds else last_speed
        return np.array([weight / 1000.0, count / 10.0, last_speed / 200.0, avg_speed / 200.0], dtype=np.float32)

    def get_reward(self, weight, count, speed, last_speed):
        if weight > self.overload_threshold or self.temperature > self.temp_threshold:
            return -50
        speed_change = abs(speed - last_speed)
        if speed_change > SPEED_CHANGE_LIMIT:
            return -30
        recent_speeds = self.rl_speeds[-SMOOTHNESS_WINDOW:] if len(self.rl_speeds) >= SMOOTHNESS_WINDOW else self.rl_speeds
        avg_speed = np.mean(recent_speeds) if recent_speeds else last_speed
        smoothness_reward = -abs(speed - avg_speed) / 25.0
        target_speed = self.normal_control(count, weight)
        target_reward = -abs(speed - target_speed) / 25.0
        if count > 8:
            if speed == 50:
                return 30 + smoothness_reward + target_reward
            if speed == 75:
                return 20 + smoothness_reward + target_reward
            return -15 + smoothness_reward + target_reward
        if count > 6:
            if speed == 75:
                return 25 + smoothness_reward + target_reward
            if speed == 100:
                return 15 + smoothness_reward + target_reward
            return -10 + smoothness_reward + target_reward
        if count > 4:
            if speed == 100:
                return 20 + smoothness_reward + target_reward
            if speed == 125:
                return 15 + smoothness_reward + target_reward
            return -10 + smoothness_reward + target_reward
        if count > 2:
            if speed == 175:
                return 15 + smoothness_reward + target_reward
            if speed == 180:
                return 10 + smoothness_reward + target_reward
            return -5 + smoothness_reward + target_reward
        if count > 1:
            if speed == 180:
                return 15 + smoothness_reward + target_reward
            if speed == 200:
                return 10 + smoothness_reward + target_reward
            return -5 + smoothness_reward + target_reward
        if count > 0:
            if speed == 200:
                return 15 + smoothness_reward + target_reward
            if speed == 175:
                return 10 + smoothness_reward + target_reward
            return -5 + smoothness_reward + target_reward
        if count == 0:
            if weight > 800:
                if speed == 175:
                    return 20 + smoothness_reward + target_reward
                if speed == 180:
                    return 15 + smoothness_reward + target_reward
                return -5 + smoothness_reward + target_reward
            if weight > 600:
                if speed == 180:
                    return 15 + smoothness_reward + target_reward
                if speed == 190:
                    return 10 + smoothness_reward + target_reward
                return -5 + smoothness_reward + target_reward
            if weight <= 50:
                if speed == 200:
                    return 20 + smoothness_reward + target_reward
                if speed in [175, 190]:
                    return 15 + smoothness_reward + target_reward
                return 5 + smoothness_reward + target_reward
            if speed in [190, 200]:
                return 15 + smoothness_reward + target_reward
            if speed == 175:
                return 10 + smoothness_reward + target_reward
            return 5 + smoothness_reward + target_reward
        return 10 + smoothness_reward + target_reward

    def normal_control(self, count, weight):
        if weight > self.overload_threshold or self.temperature > self.temp_threshold:
            return 0
        if count > 8:
            target_speed = 50
        elif count > 6:
            target_speed = 75
        elif count > 4:
            target_speed = 100
        elif count > 2:
            target_speed = 175
        elif count > 1:
            target_speed = 180
        elif count > 0:
            target_speed = 200
        else:
            if weight > 800:
                target_speed = 175
            elif weight > 600:
                target_speed = 180
            elif weight <= 50:
                target_speed = 200
            else:
                target_speed = 190
        target_speed = max(self.last_speed - SPEED_CHANGE_LIMIT,
                          min(self.last_speed + SPEED_CHANGE_LIMIT, target_speed))
        return target_speed

    def fuzzy_control(self, count, weight):
        if weight > self.overload_threshold or self.temperature > self.temp_threshold:
            return 0
        try:
            self.fuzzy_sim.input['weight'] = weight
            self.fuzzy_sim.input['count'] = count
            self.fuzzy_sim.compute()
            speed = self.fuzzy_sim.output['speed']
            speed = max(20, min(200, speed))
            speed = max(self.last_speed - SPEED_CHANGE_LIMIT,
                       min(self.last_speed + SPEED_CHANGE_LIMIT, speed))
            return speed
        except Exception as e:
            self.alert_label.setText(f"⚠️ Lỗi Fuzzy: {e}")
            logging.error(f"Fuzzy control error: {e}")
            return 0

    def dqn_control(self, count, weight):
        if weight > self.overload_threshold or self.temperature > self.temp_threshold:
            return 0
        if weight == 0 and count == 0:
            return 200
        state = self.get_state(weight, count, self.last_speed)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        valid_actions = [a for a in ACTIONS if abs(a - self.last_speed) <= SPEED_CHANGE_LIMIT]
        valid_action_indices = [ACTIONS.index(a) for a in valid_actions]
        if not valid_actions:
            valid_actions = [self.last_speed]
            valid_action_indices = [ACTIONS.index(self.last_speed)]
        if random.uniform(0, 1) < self.epsilon:
            action = random.choice(valid_actions)
        else:
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)[0][valid_action_indices]
                weights = [1.0 / (1.0 + abs(a - self.last_speed) / 25.0) for a in valid_actions]
                weighted_q_values = q_values * torch.tensor(weights, device=self.device)
                action_idx = valid_action_indices[weighted_q_values.argmax().item()]
                action = ACTIONS[action_idx]
        reward = self.get_reward(weight, count, action, self.last_speed)
        next_state = self.get_state(weight, count, action)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        self.replay_buffer.append((state, ACTIONS.index(action), reward, next_state, False))
        if len(self.replay_buffer) >= BATCH_SIZE:
            batch = random.sample(self.replay_buffer, BATCH_SIZE)
            states, actions, rewards, next_states, dones = zip(*batch)
            states = torch.FloatTensor(states).to(self.device)
            actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
            rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
            next_states = torch.FloatTensor(next_states).to(self.device)
            dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
            current_q_values = self.policy_net(states).gather(1, actions)
            with torch.no_grad():
                next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
                target_q_values = rewards + (1 - dones) * GAMMA * next_q_values
            loss = nn.MSELoss()(current_q_values, target_q_values)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        if self.step_count % TARGET_UPDATE_FREQ == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        self.epsilon = max(MIN_EPSILON, self.epsilon * EPSILON_DECAY)
        current_time = time.time() - self.start_time
        self.rl_times.append(current_time)
        self.rl_rewards.append(reward)
        self.rl_speeds.append(action)
        self.rl_weights.append(weight)
        self.rl_counts.append(count)
        self.rl_epsilons.append(self.epsilon)
        self.step_count += 1
        if len(self.rl_times) > MAX_POINTS:
            self.rl_times.pop(0)
            self.rl_rewards.pop(0)
            self.rl_speeds.pop(0)
            self.rl_weights.pop(0)
            self.rl_counts.pop(0)
            self.rl_epsilons.pop(0)
        self.update_rl_plot()
        return action

    def init_websocket(self):
        try:
            self.ws_client = websocket.WebSocket()
            self.ws_client.connect(WEBSOCKET_URL)
            threading.Thread(target=self.read_websocket_data, daemon=True).start()
            logging.info(f"Connected to WebSocket server at {WEBSOCKET_URL}")
        except Exception as e:
            logging.error(f"WebSocket connection error: {e}")
            self.alert_label.setText("⚠️ Không kết nối được với WebSocket server!")
            self.update_dot(self.alert_dot, "#FF0000")

    def send_to_websocket(self, data):
        if self.ws_client and self.ws_client.connected:
            try:
                self.ws_client.send(json.dumps(data))
                logging.info(f"Sent to WebSocket: {data}")
            except Exception as e:
                logging.error(f"WebSocket send error: {e}")
                self.alert_label.setText("⚠️ Mất kết nối với WebSocket server!")
                self.update_dot(self.alert_dot, "#FF0000")
                self.init_websocket()

    def read_websocket_data(self):
        while True:
            try:
                message = self.ws_client.recv()
                data = json.loads(message)
                command = data.get('command')
                if command == 'ALERT':
                    alert_message = data.get('message', 'Cảnh báo thủ công từ web')
                    self.alert_label.setText(f"⚠️ {alert_message}")
                    self.update_dot(self.alert_dot, "#FF0000")
                    self.save_to_db(self.weight, self.total_count, self.last_sent_speed, self.temperature, self.motor_enabled, alert_message)
                    logging.info(f"Received manual alert: {alert_message}")
                elif command:
                    with open(CONTROL_FILE, 'w') as f:
                        f.write(command)
                    logging.info(f"Received command from WebSocket: {command}")
            except Exception as e:
                logging.error(f"WebSocket receive error: {e}")
                self.init_websocket()
                break

    def toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def show_about(self):
        QtWidgets.QMessageBox.about(self, "Giới thiệu", 
                                  "Hệ thống Điều khiển Động cơ Công nghiệp\n\n"
                                  "Phiên bản: 1.0\n"
                                  "Phát triển bởi: Your Company\n"
                                  "Năm: 2025")

    def export_to_csv(self):
        try:
            with open(CSV_FILE, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Timestamp', 'Weight (g)', 'Temperature (°C)', 'Speed (RPM)', 'Count', 'Motor Status'])
                for data in self.history_data:
                    writer.writerow([
                        data['timestamp'],
                        data['weight'],
                        data['temperature'],
                        data['speed'],
                        data.get('count', 0),
                        'ON' if data.get('motor_status', False) else 'OFF'
                    ])
            QtWidgets.QMessageBox.information(self, "Thành công", f"Dữ liệu đã được xuất ra {CSV_FILE}")
            logging.info(f"Data exported to {CSV_FILE}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Lỗi", f"Không thể xuất CSV: {e}")
            logging.error(f"CSV export error: {e}")

    def init_database(self):
        try:
            self.db = mysql.connector.connect(
                host="localhost",
                user="root",
                password="",
                database="motor_control"
            )
            self.cursor = self.db.cursor(dictionary=True)
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS sensor_data (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    timestamp DATETIME,
                    weight FLOAT,
                    count INT,
                    speed FLOAT,
                    temperature FLOAT,
                    motor_status BOOLEAN
                )
            """)
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    timestamp DATETIME,
                    message TEXT,
                    level VARCHAR(20)
                )
            """)
            self.db.commit()
            logging.info("Database initialized successfully")
        except mysql.connector.Error as err:
            self.alert_label.setText(f"⚠️ Lỗi MySQL: {err}")
            self.update_dot(self.alert_dot, "#FF0000")
            logging.error(f"MySQL error: {err}")
            sys.exit(1)

    def init_serial(self):
        try:
            self.serial = serial.Serial(self.serial_port, 115200, timeout=SERIAL_TIMEOUT)
            self.serial_connected = True
            logging.info(f"Serial port {self.serial_port} initialized successfully")
        except serial.SerialException as e:
            self.serial_connected = False
            self.alert_label.setText("⚠️ Không kết nối được với ESP32!")
            self.update_dot(self.alert_dot, "#FF0000")
            logging.error(f"Serial connection error: {e}")

    def reconnect_serial(self):
        if not self.serial_connected:
            try:
                self.serial = serial.Serial(self.serial_port, 115200, timeout=SERIAL_TIMEOUT)
                self.serial_connected = True
                self.alert_label.setText("")
                self.update_dot(self.alert_dot, "#000000")
                logging.info(f"Reconnected to serial port {self.serial_port}")
                for cmd, timestamp in list(self.pending_commands.items()):
                    if time.time() - timestamp < 5:
                        self.send_to_esp32(cmd, retry=True)
            except serial.SerialException as e:
                logging.error(f"Serial reconnect failed: {e}")

    def init_camera(self):
        try:
            if self.camera_port.isdigit():
                camera_index = int(self.camera_port)
                if camera_index < 0:
                    raise ValueError("Cổng camera không hợp lệ!")
                self.capture = cv2.VideoCapture(camera_index)
                if not self.capture.isOpened():
                    for i in range(3):
                        self.capture = cv2.VideoCapture(i)
                        if self.capture.isOpened():
                            self.camera_port = str(i)
                            self.alert_label.setText(f"Đã kết nối với camera cổng {i}")
                            logging.info(f"Connected to camera at index {i}")
                            return
                    raise ValueError(f"Không thể mở camera tại cổng {self.camera_port}!")
            else:
                self.capture = cv2.VideoCapture(self.camera_port)
                if not self.capture.isOpened():
                    raise ValueError(f"Không thể mở camera tại URL {self.camera_port}!")
            logging.info(f"Camera initialized at {self.camera_port}")
        except (ValueError, cv2.error) as e:
            self.alert_label.setText(f"⚠️ Lỗi camera: {str(e)}")
            self.update_dot(self.alert_dot, "#FF0000")
            logging.error(f"Camera initialization error: {e}")
            self.capture = None
            QtWidgets.QMessageBox.critical(self, "Lỗi Camera", f"Không thể khởi tạo camera: {str(e)}. Ứng dụng sẽ chạy mà không có video.")

    def load_history_data(self):
        try:
            query = """
                SELECT timestamp, weight, temperature, speed, count, motor_status 
                FROM sensor_data 
                WHERE timestamp >= NOW() - INTERVAL %s HOUR
                ORDER BY timestamp
            """
            self.cursor.execute(query, (HISTORY_HOURS,))
            self.history_data = self.cursor.fetchall()
            logging.info("History data loaded successfully")
        except mysql.connector.Error as err:
            self.alert_label.setText(f"⚠️ Lỗi tải lịch sử: {err}")
            self.update_dot(self.alert_dot, "#FF0000")
            logging.error(f"History data load error: {err}")

    def update_history(self):
        try:
            query = """
                SELECT timestamp, weight, temperature, speed, count, motor_status 
                FROM sensor_data 
                WHERE timestamp >= %s
                ORDER BY timestamp
            """
            last_time = self.history_data[-1]['timestamp'] if self.history_data else datetime.now()
            self.cursor.execute(query, (last_time,))
            new_data = self.cursor.fetchall()
            self.history_data.extend(new_data)
            if len(self.history_data) > HISTORY_HOURS * 60:
                self.history_data = self.history_data[-HISTORY_HOURS*60:]
            self.update_history_plot()
            logging.info("History data updated")
        except mysql.connector.Error as err:
            self.alert_label.setText(f"⚠️ Lỗi cập nhật lịch sử: {err}")
            self.update_dot(self.alert_dot, "#FF0000")
            logging.error(f"History update error: {err}")

    def init_plots(self):
        self.normal_fig = plt.Figure(figsize=(5, 4))
        self.normal_canvas = FigureCanvas(self.normal_fig)
        normal_layout = self.findChild(QtWidgets.QVBoxLayout, "normalTabLayout")
        normal_layout.addWidget(self.normal_canvas)
        self.normal_ax = self.normal_fig.add_subplot(111)
        self.normal_ax.set_xlabel('Thời gian (s)')
        self.normal_ax.set_ylabel('Giá trị')
        self.normal_ax.set_title('Điều khiển Normal - Thời gian thực')
        self.normal_canvas.draw()

        self.fuzzy_fig = plt.Figure(figsize=(5, 4))
        self.fuzzy_canvas = FigureCanvas(self.fuzzy_fig)
        fuzzy_layout = self.findChild(QtWidgets.QVBoxLayout, "fuzzyTabLayout")
        fuzzy_layout.addWidget(self.fuzzy_canvas)
        self.fuzzy_ax = self.fuzzy_fig.add_subplot(111)
        self.fuzzy_ax.set_xlabel('Thời gian (s)')
        self.fuzzy_ax.set_ylabel('Giá trị')
        self.fuzzy_ax.set_title('Điều khiển Fuzzy - Thời gian thực')
        self.fuzzy_canvas.draw()

        self.rl_fig = plt.Figure(figsize=(5, 4))
        self.rl_canvas = FigureCanvas(self.rl_fig)
        rl_layout = self.findChild(QtWidgets.QVBoxLayout, "rlTabLayout")
        rl_layout.addWidget(self.rl_canvas)
        self.rl_ax = self.rl_fig.add_subplot(111)
        self.rl_ax.set_xlabel('Thời gian (s)')
        self.rl_ax.set_ylabel('Điểm thưởng trung bình / Phần thưởng', color='g')
        self.rl_ax_speed = self.rl_ax.twinx()
        self.rl_ax_speed.set_ylabel('Tốc độ (RPM) / Epsilon (x200)', color='b')
        self.rl_ax.set_title('Tiến trình học RL')
        self.rl_canvas.draw()

        self.history_fig = plt.Figure(figsize=(5, 4))
        self.history_canvas = FigureCanvas(self.history_fig)
        history_layout = self.findChild(QtWidgets.QVBoxLayout, "historyTabLayout")
        history_layout.addWidget(self.history_canvas)
        self.history_ax = self.history_fig.add_subplot(111)
        self.history_ax.set_xlabel('Thời gian')
        self.history_ax.set_ylabel('Giá trị')
        self.history_ax.set_title('Dữ liệu lịch sử')
        self.update_history_plot()
        self.history_canvas.draw()

    def update_history_plot(self):
        if not self.history_data:
            return
        times = [d['timestamp'] for d in self.history_data]
        weights = [d['weight'] for d in self.history_data]
        temps = [d['temperature'] for d in self.history_data]
        speeds = [d['speed'] for d in self.history_data]
        self.history_ax.clear()
        self.history_ax.plot(times, weights, 'b-', label='Trọng lượng (g)')
        self.history_ax.plot(times, temps, 'r-', label='Nhiệt độ (°C)')
        self.history_ax.plot(times, speeds, 'g-', label='Tốc độ (RPM)')
        self.history_ax.set_xlabel('Thời gian')
        self.history_ax.set_ylabel('Giá trị')
        self.history_ax.set_title('Dữ liệu lịch sử')
        self.history_ax.legend()
        self.history_ax.grid(True)
        for label in self.history_ax.get_xticklabels():
            label.set_rotation(45)
            label.set_ha('right')
        self.history_fig.tight_layout()
        self.history_canvas.draw()

    def save_to_db(self, weight, count, speed, temperature, motor_status, alert_message=""):
        try:
            query = """
                INSERT INTO sensor_data (timestamp, weight, count, speed, temperature, motor_status)
                VALUES (NOW(), %s, %s, %s, %s, %s)
            """
            self.cursor.execute(query, (weight, count, speed, temperature, motor_status))
            if alert_message:
                query_alert = """
                    INSERT INTO alerts (timestamp, message, level)
                    VALUES (NOW(), %s, %s)
                """
                level = "critical" if "quá tải" in alert_message.lower() else "warning"
                self.cursor.execute(query_alert, (alert_message, level))
            self.db.commit()
            logging.info("Data saved to database")
        except mysql.connector.Error as err:
            self.alert_label.setText(f"⚠️ Lỗi MySQL: {err}")
            self.update_dot(self.alert_dot, "#FF0000")
            logging.error(f"MySQL save error: {err}")

    def reset_data(self):
        try:
            self.cursor.execute("TRUNCATE TABLE sensor_data")
            self.cursor.execute("TRUNCATE TABLE alerts")
            self.db.commit()
            self.normal_times.clear()
            self.normal_weights.clear()
            self.normal_counts.clear()
            self.normal_speeds.clear()
            self.fuzzy_times.clear()
            self.fuzzy_weights.clear()
            self.fuzzy_counts.clear()
            self.fuzzy_speeds.clear()
            self.rl_times.clear()
            self.rl_rewards.clear()
            self.rl_speeds.clear()
            self.rl_weights.clear()
            self.rl_counts.clear()
            self.rl_epsilons.clear()
            self.history_data.clear()
            self.normal_ax.clear()
            self.fuzzy_ax.clear()
            self.rl_ax.clear()
            self.rl_ax_speed.clear()
            self.history_ax.clear()
            self.normal_ax.set_xlabel('Thời gian (s)')
            self.normal_ax.set_ylabel('Giá trị')
            self.normal_ax.set_title('Điều khiển Normal - Thời gian thực')
            self.fuzzy_ax.set_xlabel('Thời gian (s)')
            self.fuzzy_ax.set_ylabel('Giá trị')
            self.fuzzy_ax.set_title('Điều khiển Fuzzy - Thời gian thực')
            self.rl_ax.set_xlabel('Thời gian (s)')
            self.rl_ax.set_ylabel('Điểm thưởng trung bình / Phần thưởng', color='g')
            self.rl_ax_speed.set_ylabel('Tốc độ (RPM) / Epsilon (x200)', color='b')
            self.rl_ax.set_title('Tiến trình học RL')
            self.history_ax.set_xlabel('Thời gian')
            self.history_ax.set_ylabel('Giá trị')
            self.history_ax.set_title('Dữ liệu lịch sử')
            self.normal_canvas.draw()
            self.fuzzy_canvas.draw()
            self.rl_canvas.draw()
            self.history_canvas.draw()
            self.count_label.setText("Số lượng: 0")
            self.weight_label.setText("0.00 g")
            self.speed_label.setText("0 RPM")
            self.temp_label.setText("0.0 °C")
            self.detail_label.setText("Chi tiết:\n" + "\n".join([f"{lbl}: 0" for lbl in self.class_labels]))
            for label in self.class_labels:
                self.class_counts[label] = 0
            self.speed_gauge.update_value(0)
            self.temp_gauge.update_value(0)
            self.weight_gauge.update_value(0)
            self.last_speed = 100
            logging.info("Data reset successfully")
        except mysql.connector.Error as err:
            self.alert_label.setText(f"⚠️ Lỗi reset MySQL: {err}")
            self.update_dot(self.alert_dot, "#FF0000")
            logging.error(f"MySQL reset error: {err}")

    def update_normal_fuzzy_plot(self, mode, weight, count, speed):
        current_time = time.time() - self.start_time
        if mode == "Normal":
            times, weights, counts, speeds, ax, canvas = (
                self.normal_times, self.normal_weights, self.normal_counts, 
                self.normal_speeds, self.normal_ax, self.normal_canvas
            )
        else:
            times, weights, counts, speeds, ax, canvas = (
                self.fuzzy_times, self.fuzzy_weights, self.fuzzy_counts, 
                self.fuzzy_speeds, self.fuzzy_ax, self.fuzzy_canvas
            )
        times.append(current_time)
        weights.append(weight)
        counts.append(count)
        speeds.append(speed)
        if len(times) > MAX_POINTS:
            times.pop(0)
            weights.pop(0)
            counts.pop(0)
            speeds.pop(0)
        ax.clear()
        ax.plot(times, weights, 'b-', label='Trọng lượng (g)')
        ax.plot(times, counts, 'r-', label='Số lượng')
        ax.plot(times, speeds, 'g-', label='Tốc độ (RPM)')
        ax.set_xlabel('Thời gian (s)')
        ax.set_ylabel('Giá trị')
        ax.set_title(f'{mode} Control - Thời gian thực')
        ax.legend()
        ax.grid(True)
        canvas.draw()

    def update_rl_plot(self):
        if len(self.rl_rewards) < 2:
            return
        window_size = 10
        avg_rewards = [np.mean(self.rl_rewards[max(0, i - window_size):i + 1]) for i in range(len(self.rl_rewards))]
        min_length = min(len(self.rl_times), len(self.rl_rewards), len(self.rl_speeds), len(self.rl_epsilons))
        if min_length == 0:
            return
        times = self.rl_times[:min_length]
        rewards = self.rl_rewards[:min_length]
        speeds = self.rl_speeds[:min_length]
        epsilons = self.rl_epsilons[:min_length]
        self.rl_ax.clear()
        self.rl_ax_speed.clear()
        self.rl_ax.plot(times, avg_rewards, 'g-', label='Điểm thưởng trung bình')
        self.rl_ax.plot(times, rewards, 'y-', label='Phần thưởng', alpha=0.5)
        self.rl_ax.set_xlabel('Thời gian (s)')
        self.rl_ax.set_ylabel('Điểm thưởng trung bình / Phần thưởng', color='g')
        self.rl_ax.tick_params(axis='y', labelcolor='g')
        self.rl_ax_speed.plot(times, speeds, 'b-', label='Tốc độ (RPM)')
        self.rl_ax_speed.plot(times, [e * 200 for e in epsilons], 'm-', label='Epsilon (x200)', alpha=0.5)
        self.rl_ax_speed.set_ylabel('Tốc độ (RPM) / Epsilon (x200)', color='b')
        self.rl_ax_speed.tick_params(axis='y', labelcolor='b')
        self.rl_ax.set_title('Tiến trình học RL')
        self.rl_ax.legend(loc='upper left')
        self.rl_ax_speed.legend(loc='upper right')
        self.rl_ax.grid(True)
        self.rl_canvas.draw()

    def update_dot(self, label, color):
        label.setStyleSheet(f"background-color: {color}; border-radius: 10px; border: 1px solid black;")

    def toggle_motor(self):
        self.motor_enabled = not self.motor_enabled
        cmd = f"ENABLE:{1 if self.motor_enabled else 0}"
        self.send_to_esp32(cmd)
        self.pending_commands[cmd] = time.time()
        self.toggle_button.setText("Tắt động cơ" if self.motor_enabled else "Bật động cơ")
        self.update_dot(self.motor_dot, "#FFFF00" if self.motor_enabled else "#000000")
        self.motor_status_label.setText("Động cơ: Bật" if self.motor_enabled else "Động cơ: Tắt")
        if not self.motor_enabled:
            self.speed_label.setText("0 RPM")
            self.speed_gauge.update_value(0)
            self.last_sent_speed = 0
            self.last_speed = 0
            self.send_to_esp32("SPEED:0")
        logging.info(f"Motor toggled: {'Enabled' if self.motor_enabled else 'Disabled'}")

    def read_control_file(self):
        if CONTROL_FILE.exists():
            with open(CONTROL_FILE, 'r') as f:
                command = f.read().strip()
            if command == 'TOGGLE':
                self.toggle_motor()
            elif command == 'RESET':
                self.reset_data()
            elif command.startswith('SET_SPEED:'):
                try:
                    speed = int(command.split(':')[1])
                    if 0 <= speed <= 200:
                        if self.motor_enabled:
                            cmd = f"SPEED:{speed}"
                            self.send_to_esp32(cmd)
                            self.pending_commands[cmd] = time.time()
                            self.last_sent_speed = speed
                            self.last_speed = speed
                            self.speed_label.setText(f"{speed:.1f} RPM")
                            self.speed_gauge.update_value(speed)
                            logging.info(f"Set speed to {speed} RPM from control file")
                        else:
                            logging.warning("Cannot set speed: Motor is disabled")
                except (IndexError, ValueError):
                    logging.error("Invalid SET_SPEED command")
            CONTROL_FILE.unlink(missing_ok=True)

    def update_frame(self):
        if not hasattr(self, 'capture') or self.capture is None or not self.capture.isOpened():
            self.alert_label.setText("⚠️ Không có camera khả dụng!")
            logging.warning("No camera available")
            return
        ret, frame = self.capture.read()
        if not ret:
            self.alert_label.setText("⚠️ Không thể đọc khung hình từ camera!")
            logging.warning("Failed to read frame from camera")
            return
        
        frame = cv2.resize(frame, (640, 480))
        self.frame_count += 1
        if self.frame_count % 3 == 0:
            results = self.model(frame)
            self.last_results = results
        else:
            results = self.last_results if self.last_results is not None else []
        
        output = frame.copy()
        current_centers = []
        cv2.polylines(output, [self.roi_polygon], isClosed=True, color=(0, 255, 255), thickness=2)
        cv2.line(output, self.line_start, self.line_end, (0, 0, 255), 2)
        current_count = 0
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                center = (cx, cy)
                cls_id = int(box.cls)
                label = self.label_names[cls_id]
                conf = float(box.conf)
                
                if cv2.pointPolygonTest(self.roi_polygon, center, False) >= 0:
                    current_count += 1
                    cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label_text = f"{label} {conf:.2f}"
                    cv2.putText(output, label_text, (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    for prev in self.prev_centers:
                        pcx, pcy = prev
                        if abs(pcx - cx) < 30 and pcy > self.line_y >= cy:
                            self.class_counts[label] += 1
                            break
                    current_centers.append(center)
        
        self.total_count = current_count
        self.prev_centers = current_centers.copy()
        self.count_label.setText(f"Số lượng: {self.total_count}")
        
        while not self.serial_queue.empty():
            key, value = self.serial_queue.get()
            if key == "WEIGHT":
                self.weight = float(value)
                self.weight_label.setText(f"{self.weight:.2f} g")
                self.weight_gauge.update_value(self.weight)
            elif key == "TEMP":
                self.temperature = float(value)
                self.temp_label.setText(f"{self.temperature:.1f} °C")
                self.temp_gauge.update_value(self.temperature)
            elif key == "BTN":
                btn_state = value == "1"
                if btn_state != self.motor_enabled:
                    self.motor_enabled = btn_state
                    self.toggle_button.setText("Tắt động cơ" if self.motor_enabled else "Bật động cơ")
                    self.update_dot(self.motor_dot, "#FFFF00" if self.motor_enabled else "#000000")
                    self.motor_status_label.setText("Động cơ: Bật" if self.motor_enabled else "Động cơ: Tắt")
                    if not self.motor_enabled:
                        self.speed_label.setText("0 RPM")
                        self.speed_gauge.update_value(0)
                        self.last_sent_speed = 0
                        self.last_speed = 0
                        self.send_to_esp32("SPEED:0")
                    if f"ENABLE:{1 if btn_state else 0}" in self.pending_commands:
                        del self.pending_commands[f"ENABLE:{1 if btn_state else 0}"]
            elif key == "SPEED":
                speed = float(value)
                if speed != self.last_sent_speed:
                    self.speed_label.setText(f"{speed:.1f} RPM")
                    self.speed_gauge.update_value(speed)
                    self.last_sent_speed = speed
                    self.last_speed = speed
                    logging.info(f"Synchronized SPEED from ESP32: {speed}")
                if f"SPEED:{speed}" in self.pending_commands:
                    del self.pending_commands[f"SPEED:{speed}"]
            elif key == "ACK":
                if value in self.pending_commands:
                    del self.pending_commands[value]
                    logging.info(f"ACK received for {value}")

        self.temp_label.setText(f"{self.temperature:.1f} °C")
        detail_text = 'Chi tiết:\n' + '\n'.join([f"{lbl}: {cnt}" for lbl, cnt in self.class_counts.items()])
        self.detail_label.setText(detail_text)
        alert_message = ""
        if self.weight > self.overload_threshold or self.temperature > self.temp_threshold:
            alert_message = "Cảnh báo: Quá tải hoặc nhiệt độ cao!"
            self.alert_label.setText("⚠️ Quá tải hoặc nhiệt độ cao!")
            self.send_to_esp32("SPEED:0")
            self.update_dot(self.alert_dot, "#FF0000")
            self.motor_enabled = False
            self.toggle_button.setText("Bật động cơ")
            self.update_dot(self.motor_dot, "#000000")
            self.motor_status_label.setText("Động cơ: Tắt")
            self.last_sent_speed = 0
            self.last_speed = 0
            self.speed_label.setText("0 RPM")
            self.speed_gauge.update_value(0)
            logging.warning("Overload or high temperature detected")
        else:
            self.alert_label.setText("")
            self.update_dot(self.alert_dot, "#000000")
            if self.motor_enabled:
                if self.control_mode == "Normal":
                    speed = self.normal_control(self.total_count, self.weight)
                elif self.control_mode == "Fuzzy":
                    speed = self.fuzzy_control(self.total_count, self.weight)
                else:
                    speed = self.dqn_control(self.total_count, self.weight)
                if speed != self.last_sent_speed:
                    cmd = f"SPEED:{speed}"
                    self.send_to_esp32(cmd)
                    self.pending_commands[cmd] = time.time()
                    self.last_sent_speed = speed
                    self.last_speed = speed
                    self.speed_label.setText(f"{speed:.1f} RPM")
                    self.speed_gauge.update_value(speed)
                if self.control_mode in ["Normal", "Fuzzy"]:
                    self.update_normal_fuzzy_plot(self.control_mode, self.weight, self.total_count, speed)
                else:
                    self.update_normal_fuzzy_plot("RL", self.weight, self.total_count, speed)
            else:
                speed = 0
                if self.last_sent_speed != 0:
                    cmd = "SPEED:0"
                    self.send_to_esp32(cmd)
                    self.pending_commands[cmd] = time.time()
                    self.last_sent_speed = 0
                    self.last_speed = 0
                    self.speed_label.setText("0 RPM")
                    self.speed_gauge.update_value(0)
        
        data = {
            "timestamp": datetime.now().isoformat(),
            "weight": float(self.weight),
            "count": int(self.total_count),
            "speed": float(self.last_sent_speed),
            "temperature": float(self.temperature),
            "motor_status": bool(self.motor_enabled),
            "alert": str(self.alert_label.text()),
            "class_counts": {k: int(v) for k, v in self.class_counts.items()}
        }
        self.send_to_websocket(data)
        
        self.save_to_db(self.weight, self.total_count, speed, self.temperature, self.motor_enabled, alert_message)
        rgb_image = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        qt_image = QtGui.QImage(rgb_image.data, w, h, ch * w, QtGui.QImage.Format.Format_RGB888)
        self.video_label.setPixmap(QtGui.QPixmap.fromImage(qt_image).scaled(
            self.video_label.width(), self.video_label.height(), Qt.AspectRatioMode.KeepAspectRatio))
        if self.control_mode == "RL":
            torch.save(self.policy_net.state_dict(), DQN_MODEL_FILE)
        logging.info("Frame updated")

    def send_to_esp32(self, msg, retry=False):
        if self.serial_connected and hasattr(self, 'serial') and self.serial.is_open:
            try:
                self.serial.write((msg + "\n").encode())
                self.serial.flush()
                if not retry:
                    self.pending_commands[msg] = time.time()
                logging.info(f"Sent to ESP32: {msg}")
            except serial.SerialException as e:
                self.serial_connected = False
                self.alert_label.setText("⚠️ Mất kết nối với ESP32!")
                self.update_dot(self.alert_dot, "#FF0000")
                logging.error(f"ESP32 send error: {e}")
                self.reconnect_serial()

    def read_serial_data(self):
        while True:
            if not self.serial_connected or not hasattr(self, 'serial') or not self.serial.is_open:
                self.reconnect_serial()
                time.sleep(2)
                continue
            try:
                if self.serial.in_waiting > 0:
                    line = self.serial.readline().decode('utf-8', errors='ignore').strip()
                    if line.startswith("<") and line.endswith(">"):
                        clean = line[1:-1]
                        parts = clean.split(',')
                        for part in parts:
                            try:
                                key, value = part.split(":")
                                if key in ["WEIGHT", "TEMP", "BTN", "SPEED", "ACK"]:
                                    self.serial_queue.put((key, value))
                                elif key != "LOG":
                                    logging.warning(f"Unknown serial key: {key}")
                            except (ValueError, IndexError):
                                logging.warning(f"Invalid serial data: {part}")
                    else:
                        logging.warning(f"Invalid serial format: {line}")
            except serial.SerialException as e:
                self.serial_connected = False
                self.alert_label.setText("⚠️ Mất kết nối với ESP32!")
                self.update_dot(self.alert_dot, "#FF0000")
                logging.error(f"Serial read error: {e}")
                self.reconnect_serial()
                time.sleep(2)

    def closeEvent(self, event):
        if hasattr(self, 'capture') and self.capture is not None:
            self.capture.release()
        if hasattr(self, 'serial') and self.serial_connected and self.serial.is_open:
            self.serial.close()
        if self.control_mode == "RL":
            torch.save(self.policy_net.state_dict(), DQN_MODEL_FILE)
        if hasattr(self, 'db') and self.db.is_connected():
            self.db.close()
        if self.ws_client and self.ws_client.connected:
            self.ws_client.close()
        logging.info("Application closed")
        event.accept()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MotorControlApp()
    window.show()
    sys.exit(app.exec())