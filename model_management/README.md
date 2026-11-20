# 模型管理接口

本包集中管理 `store.py`（持久化）、`api.py`（路由和参数校验）、
`models.html`（管理页面）和 `test_registry.py`（验证）。
训练服务仅调用 `ModelStore.register` 登记结果，`train.api` 挂载本包路由。

启动：`python start.py`，默认地址 `http://127.0.0.1:8008`。
Swagger：`/docs`；Apifox 可导入项目的 `openapi.json`。

| 方法 | 路径 | 作用 |
|---|---|---|
| GET | /model-registry?page=1&page_size=10 | 分页模型列表 |
| GET | /model-registry/{id} | 模型详情和完整 test 结果 |
| PATCH | /model-registry/{id} | 修改名称、用途 |
| DELETE | /model-registry/{id} | 软删除 |
| POST | /model-registry/batch-delete | 批量软删除，提交 `{"ids":["id1","id2"]}` |
| POST | /model-registry/delete-all | 软删除全部可用模型，包含所有分页，无需请求体 |

批量接口最多接收 100 个 ID，自动去重，返回 `deleted` 和 `failed` 两个列表。
每个模型复用单项删除逻辑；部分失败时其余模型继续处理，不永久删除模型文件。
普通勾选只删除当前页所选模型，翻页或刷新会清空勾选。
点击表头“全选”表示跨页全部选择，确认按钮显示总数并调用独立的全部删除接口；
手动勾选本页每一行仍是单页删除，不会触发全部删除。失败项仍在当前页时保留勾选以便重试。

模型由现有 `POST /train` 在训练和 test 成功后自动入表，无需前端另发新增请求。
数据库位于 `models/registry.sqlite3`，重启后保留。

`POST /train` 必须同时上传 `train_data`、`test_data` 和 `label_mapping`。
映射文件支持 UTF-8 JSON 对象（如 `{"0":"体育","1":"财经"}`），
或 TXT/TSV 每行 `label<TAB>类别名称`，不带表头。
label 必须与训练集、测试集中的原始标签对应，类别名称和 label 均不可重复或为空。
映射文件替代原来的 `labels` 表单字段，同步和异步训练均使用该映射，
最终模型目录中保存 `label_mapping.json`。当前客服意图数据集的映射示例位于
`datasets/downloads/label_mapping.json`，与 `train_set.jsonl` 同目录。

fastText 支持手动训练和自动调参。前端选择 fastText 后勾选“自动调参”，
设置搜索时长（默认 300 秒）；字符级/词级仍需手动选择。
接口 `params` 示例：`{"tokenizer":"word","autotune":true,"autotune_duration":300}`。
自动模式使用对应的 dev 验证文件及 fastText 原生 F1 搜索目标，不使用 test 文件选参数；
手动超参数不传入搜索接口，避免将搜索参数锁定。
最终参数保存在模型目录的 `training_config.json`，并写入日志和模型登记参数。
搜索预算不包含最终最佳模型重新训练及评估的全部耗时。

训练产物保存到 `models/{模型类型}/{模型名}/`，每次训练使用独立的同名目录。
仅登记训练、测试成功后的最终模型，RF 和 fastText 各保存一份权重；
BERT 的 `final/` 仅保留一份验证集最优权重及必要配置、分词器，不保留中间检查点。

## 训练用途（必填）

`POST /train` 的表单字段 `purpose` 必填，只接受用途枚举的英文标识，
缺失、空值或未知标识返回 422。
调用 `GET /model-registry-options` 获取 `purposes: [{value, label}]`，
以及 `en_to_zh`、`zh_to_en` 两份字典。统一定义位于 `purposes.py`。
例如 `news` = 新闻分类、`intent` = 意图识别、`sentiment` = 情感分析，
还提供主题、工单、投诉、垃圾信息、风险、内容安全等共 14 种用途。
模型命名示例：`rf_20261119_213000_news_a1b2c3`。
内部训练服务也要求显式提供用途，可传枚举英文标识或字典中的中文名称。
管理接口修改用途仍支持业务描述，不会改变已生成的模型目录名。

## 懒加载列表

`page` 从 1 开始，`page_size` 默认 10，最大 100。按登记时间和 id 倒序。

```json
{
  "items": [],
  "page": 1,
  "page_size": 10,
  "total": 0,
  "has_more": false
}
```

每个 item 包含 `id`、`name`、`model`、`path`、`purpose`、`created_at`、
`train_source`、`test_source`、`is_deleted` 和 `test_result`。
`model` 和 `purpose` 为独立数据库字段，不依赖模型名解析。
列表、详情同时返回 `model_label` 和 `purpose_label` 中文展示名称，
例如 `model=rf`、`model_label=随机森林`、`purpose=news`、`purpose_label=新闻分类`。
列表的 `test_result` 仅含 `scores`（accuracy、precision、recall、f1 等）和 `rows`。
详情额外返回 `params` 以及完整 test 结果（分类报告、混淆矩阵、错误示例等）。

首次请求第 1 页，触底后请求下一页并追加 `items`，`has_more=false` 停止。
请求期间防止重复加载。新增、修改、删除后从第 1 页重新加载；分页期间有数据增删时，
前端可按 id 去重。页码超出范围返回空 items。

## 修改

```http
PATCH /model-registry/{id}
Content-Type: application/json

{"name":"新闻分类模型 v2","purpose":"中文新闻标题分类"}
```

两个字段至少传一个，各为 1–200 字符，自动去除首尾空白。
其他字段、null、空字符串和空请求体返回 422。
修改成功返回完整模型记录。名称与模型目录保持一致，改名会同步重命名目录并更新数据库路径。

## 删除

返回 `{"id":"模型id","is_deleted":true}`。设置软删除标记，并给数据库名称及模型目录加上 `_del_` 前缀，同步更新路径，保留模型文件。
重复删除不会重复添加前缀。
重复删除成功；不存在的 id 返回 404。
已删除记录不出现在列表中，详情和修改返回 404。

## 错误

参数校验错误为 422；模型不存在为 404，格式为 `{"detail":"模型记录不存在"}`。
其他业务参数错误为 400，未处理异常为 500。

## 推理标签映射

三类模型训练成功后，模型管理模块将训练实际使用的编号到类别名称映射保存到
`models/label/<模型名>.json`，例如 `{"0":"商品咨询","1":"退款售后"}`。
模型表新增 `label_path` 列存放绝对路径，列表及详情接口均返回该字段；模型改名、
软删除时映射文件和路径同步变更。历史记录没有映射时该列为 null。
推理读取该 JSON 后用 `mapping[str(predicted_id)]` 获取类别名称；fastText 本身返回类别名称标签。
模型目录里的 `label_mapping.json` 仍表示上传的原始标签映射，请勿将其当作模型输出编号映射。
