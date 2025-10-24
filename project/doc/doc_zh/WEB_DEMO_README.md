# Web 演示使用说明

## 快速启动

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 准备数据和模型（首次运行）

如果还没有生成数据和训练模型，需要先运行：

```bash
# 生成数据
python main.py generate_data

# 训练模型
python main.py train_model
```

### 3. 启动 Web 服务
```bash
python web_api.py
```

看到以下提示表示启动成功：
```
✓ 服务启动成功!
📍 访问地址: http://localhost:5000
```

### 4. 打开浏览器
访问：http://localhost:5000

## 功能说明

### 主要功能
1. **模拟用户行为**：点击"模拟用户行为"按钮，系统会随机选择一个用户，展示其交互历史
2. **查看推荐结果**：自动为该用户生成个性化推荐，展示推荐商品及推荐理由

### 页面展示内容
- **用户交互历史**：
  - 用户ID
  - 总交互次数
  - 交互类型分布（浏览/收藏/加购）
  - 最近的10条交互记录

- **推荐结果**：
  - Top 10 推荐商品
  - 商品分类和价格
  - 推荐评分
  - 相似商品信息
  - 推荐理由解释

## 技术架构

### 后端 (web_api.py)
- Flask Web 框架
- 提供 RESTful API 接口
- 接口列表：
  - `GET /` - 主页面
  - `POST /api/simulate_user` - 模拟用户交互
  - `GET /api/recommend?user_id=xxx` - 获取推荐
  - `GET /api/status` - 检查系统状态

### 前端 (static/index.html)
- 原生 HTML + JavaScript
- Bootstrap 5 样式框架
- 响应式设计

## 故障排查

### 问题1: 无法连接到后端服务
**解决方案**：确保 `python web_api.py` 正在运行

### 问题2: 提示"数据文件不存在"
**解决方案**：运行 `python main.py generate_data`

### 问题3: 提示"模型文件不存在"
**解决方案**：运行 `python main.py train_model`

### 问题4: 端口 5000 被占用
**解决方案**：修改 `web_api.py` 最后一行的端口号：
```python
app.run(debug=True, port=5001, host='0.0.0.0')  # 改为5001或其他端口
```

## 项目结构
```
project/
├── web_api.py              # Web API 后端
├── static/
│   └── index.html          # 前端页面
├── data/                   # 数据文件目录
│   ├── interactions.csv
│   └── items.csv
├── models/                 # 模型文件目录
│   ├── similarity_matrix.npy
│   └── item_index.pkl
└── main.py                 # 命令行工具
```

## 注意事项
1. 确保在项目根目录运行命令
2. 浏览器推荐使用 Chrome/Firefox/Edge 最新版本
3. 首次加载可能需要几秒钟时间来初始化模型

