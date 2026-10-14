"""紧凑白色工作台，参考用户提供的 UI_STYLE.md。"""

ACCENT = "#e0651c"
BG = "#ffffff"
BG_ALT = "#fafafa"
BG_PANEL = "#fafafa"
BORDER = "#eeeeee"
TEXT = "#18181a"
TEXT_DIM = "#6b6b6e"

STYLESHEET = """
QWidget { font-family: "Microsoft YaHei", "Segoe UI", sans-serif;
    font-size:13px; color:#18181a; background:#ffffff; }
QWidget#shell { background:#ffffff; }
QWidget#sidebar { background:#fafafa; border-right:1px solid #eeeeee; }
QWidget#sidebar QLabel { background:transparent; }
QLabel { background:transparent; border:none; }
QLabel[role="title"] { font-size:23px; font-weight:600; }
QLabel[role="brand"] { font-size:18px; font-weight:600; }
QLabel[role="section"] { font-size:14px; font-weight:600; }
QLabel[role="muted"] { color:#85858a; font-size:12px; }
QLabel[role="status"] { color:#6b6b6e; font-size:12px; }
QLabel[role="badge"] { color:#83907c; background:#f3f5f1;
    border-radius:4px; padding:4px 8px; font-size:12px; }
QWidget#inspector { border-left:1px solid #eeeeee; }
QWidget#runBar { border-top:1px solid #eeeeee; background:#fafafa; }
QWidget#runBar QLabel { background:transparent; }
QListWidget { background:transparent; border:none; outline:none; padding:0; }
QListWidget::item { padding:10px 12px; border-radius:6px; margin:2px 0; }
QListWidget::item:selected { background:#f1f1f1; color:#18181a; }
QListWidget::item:hover:!selected { background:#f3f3f4; }
QPushButton { background:#ffffff; border:1px solid #e3e3e5;
    border-radius:6px; padding:6px 10px; min-height:16px; }
QPushButton:hover { background:#f3f3f4; border-color:#d6d6d8; }
QPushButton:pressed { background:#eaeaec; }
QPushButton:checked { background:#f1f1f1; border-color:#d9ac8c; }
QPushButton#primary { background:#18181a; color:white; border-color:#18181a; }
QPushButton#primary:hover { background:#343436; }
QPushButton:disabled { color:#b0b0b4; background:#f7f7f7; border-color:#eeeeef; }
QPushButton#primary:disabled { background:#d6d6d9; border-color:#d6d6d9; color:white; }
QPushButton[kind="text"] { border:none; background:transparent; color:#6b6b6e; }
QPushButton[kind="text"]:hover { background:#f1f1f1; color:#18181a; }
QPushButton[kind="sideTab"] { background:#fafafa; color:#85858a; border-color:#eeeeee; }
QPushButton[kind="sideTab"]:checked { background:white; color:#18181a;
    border-color:#bdbdc1; font-weight:600; }
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox { background:white;
    border:1px solid #e5e5e7; border-radius:6px; padding:6px 8px;
    selection-background-color:#f1f1f1; selection-color:#18181a; min-height:16px; }
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus { border-color:#d9ac8c; }
QComboBox::drop-down { border:none; width:20px; }
QComboBox QAbstractItemView { background:white; border:1px solid #e5e5e7;
    selection-background-color:#f1f1f1; selection-color:#18181a; }
QComboBox:disabled, QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {
    color:#a5a5aa; background:#fafafa; border-color:#eeeeee; }
QCheckBox { spacing:6px; background:transparent; }
QCheckBox::indicator { width:14px; height:14px; }
QLabel:disabled { color:#a5a5aa; }
QTabWidget::pane { border:0; background:white; }
QTabBar::tab { background:transparent; color:#6b6b6e; padding:9px 12px;
    border-bottom:2px solid transparent; margin-right:4px; }
QTabBar::tab:selected { color:#18181a; border-bottom:2px solid #18181a; }
QTabBar::tab:hover { background:#fafafa; }
QTableWidget { background:white; border:0; gridline-color:#eeeeee;
    selection-background-color:#f1f1f1; selection-color:#18181a; outline:0; }
QTableWidget::item { padding:6px; border-bottom:1px solid #eeeeee; }
QTableWidget::item:selected { background:#f1f1f1; color:#18181a; }
QHeaderView::section { background:#fafafa; color:#85858a; border:0;
    border-bottom:1px solid #eeeeee; padding:7px 8px; font-size:12px; }
QProgressBar { background:#eaeaec; border:0; border-radius:1px;
    min-height:3px; max-height:3px; }
QProgressBar::chunk { background:#e0651c; }
QSlider::groove:horizontal { height:3px; background:#e5e5e7; border-radius:1px; }
QSlider::sub-page:horizontal { background:#e0651c; }
QSlider::handle:horizontal { width:10px; background:white; border:1px solid #d9ac8c;
    border-radius:5px; margin:-4px 0; }
QScrollArea { border:none; background:white; }
QScrollBar:vertical { background:transparent; width:8px; margin:2px; }
QScrollBar::handle:vertical { background:#d9d9dc; border-radius:3px; min-height:24px; }
QScrollBar::handle:vertical:hover { background:#bfbfc3; }
QScrollBar:horizontal { background:transparent; height:8px; margin:2px; }
QScrollBar::handle:horizontal { background:#d9d9dc; border-radius:3px; min-width:24px; }
QScrollBar::add-line, QScrollBar::sub-line { height:0; width:0; }
QScrollBar::add-page, QScrollBar::sub-page { background:transparent; }
QSplitter::handle { background:#eeeeee; width:1px; height:1px; }
QStatusBar { background:#fafafa; color:#85858a; font-size:12px; border-top:1px solid #eeeeee; }
QMenuBar { background:#fafafa; border-bottom:1px solid #eeeeee; }
QMenuBar::item:selected, QMenu::item:selected { background:#f1f1f1; }
QMenu { border:1px solid #e5e5e7; padding:4px; }
QToolTip { background:white; color:#18181a; border:1px solid #e5e5e7; padding:6px; }
QGraphicsView { border:1px solid #eeeeee; border-radius:6px; }
"""
