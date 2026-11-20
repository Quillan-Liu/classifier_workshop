"""持久化模型登记表，同步模型名称与目录名称。"""

from train.logging import log_call as _log_call

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from .purposes import MODEL_TYPE_LABELS, PURPOSE_EN_TO_ZH

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
LABELS_DIR = MODELS_DIR / "label"


class ModelStore:
    def __init__(self, path=None, label_dir=None):
        """保存 SQLite 数据库位置；省略时使用 models/registry.sqlite3。

        Args:
            path: 待读取的数据文件、目录或配置文件路径。

        Returns:
            None；数据库在首次访问时创建。
        """
        with _log_call("model_management.store.ModelStore.__init__", locals()):
            self.path = Path(path) if path is not None else MODELS_DIR / "registry.sqlite3"
            self.label_dir = Path(label_dir) if label_dir is not None else (
                LABELS_DIR if path is None else self.path.parent / "label")

    @contextmanager
    def _connect(self):
        """创建数据库连接及模型表，兼容补充软删除字段，并管理事务和连接关闭。

        Yields:
            上下文管理器，提供按列名访问记录的 sqlite3.Connection。
        """
        with _log_call("model_management.store.ModelStore._connect", locals()):
            self.path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(self.path, timeout=30)
            connection.row_factory = sqlite3.Row
            try:
                with connection:
                    connection.execute("""CREATE TABLE IF NOT EXISTS models (
                        id TEXT PRIMARY KEY, name TEXT NOT NULL, model TEXT NOT NULL,
                        path TEXT NOT NULL UNIQUE, purpose TEXT NOT NULL,
                        created_at TEXT NOT NULL, train_source TEXT NOT NULL,
                        test_source TEXT NOT NULL, params TEXT NOT NULL,
                        test_result TEXT NOT NULL)""")
                    columns = {row[1] for row in connection.execute("PRAGMA table_info(models)")}
                    if "is_deleted" not in columns:
                        connection.execute("ALTER TABLE models ADD COLUMN is_deleted INTEGER NOT NULL DEFAULT 0")
                    if "labels" not in columns:
                        connection.execute("ALTER TABLE models ADD COLUMN labels TEXT")
                    if "label_path" not in columns:
                        connection.execute("ALTER TABLE models ADD COLUMN label_path TEXT")
                    yield connection
            finally:
                connection.close()

    def register(self, *, job, model_path, params, evaluation, label_mapping=None):
        """将成功训练和测试后的模型元数据持久化，模型路径保存为绝对路径。

        Args:
            job: 训练任务对象，提供名称、类型、用途、数据来源及时间等信息。
            model_path: 模型产物目录路径。
            params: 训练参数字典；缺省字段使用模型默认配置。
            evaluation: 测试结果字典，包含 scores、rows 及完整报告。
            label_mapping: 训练实际使用的编号到类别名称映射。

        Returns:
            None；同 id 再登记只更新参数及 test 结果，路径唯一冲突由数据库报错。

        Raises:
            sqlite3.IntegrityError: 模型路径等唯一约束发生冲突。
        """
        with _log_call("model_management.store.ModelStore.register", locals()):
            values = (job.id, Path(model_path).name, job.model, str(Path(model_path).resolve()),
                      job.purpose, job.finished_at, job.source, job.test_source,
                      json.dumps(params, ensure_ascii=False),
                      json.dumps(evaluation, ensure_ascii=False))
            label_path = (self.label_dir / f"{Path(model_path).name}.json").resolve() if label_mapping is not None else None
            with self._save_labels(label_path, label_mapping), self._connect() as connection:
                connection.execute("""INSERT INTO models
                    (id,name,model,path,purpose,created_at,train_source,test_source,params,test_result,label_path)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(id) DO UPDATE SET test_result=excluded.test_result,
                    params=excluded.params, name=excluded.name, path=excluded.path,
                    label_path=COALESCE(excluded.label_path, models.label_path)""",
                    values + (str(label_path) if label_path else None,))
                if label_mapping is not None:
                    connection.execute("UPDATE models SET labels=? WHERE id=?", (json.dumps(label_mapping, ensure_ascii=False), job.id))
            self._refresh_cache()

    @contextmanager
    def _save_labels(self, path, mapping):
        """保存实际训练类别映射，登记失败时恢复原文件。"""
        if path is None:
            yield
            return
        if (not isinstance(mapping, dict) or not mapping
                or {str(key) for key in mapping} != {str(i) for i in range(len(mapping))}
                or any(not isinstance(value, str) or not value.strip() for value in mapping.values())
                or len(set(mapping.values())) != len(mapping)):
            raise ValueError("标签映射必须为从 0 开始的连续编号到唯一类别名称的非空字典")
        path.parent.mkdir(parents=True, exist_ok=True)
        previous = path.read_bytes() if path.exists() else None
        try:
            path.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
            yield
        except Exception:
            if previous is None:
                path.unlink(missing_ok=True)
            else:
                path.write_bytes(previous)
            raise

    @contextmanager
    def _rename_directory(self, old_path, name, label_path=None):
        """限制为同级目录改名；数据库操作失败时恢复目录。"""
        if (name in {".", ".."} or name.endswith((".", " "))
                or any(c in name for c in '<>:"/\\|?*')
                or any(ord(c) < 32 for c in name)):
            raise ValueError("模型名称必须是合法的文件夹名称")
        old_path = Path(old_path)
        target = old_path.with_name(name)
        old_label = Path(label_path) if label_path else None
        new_label = old_label.with_name(f"{name}.json") if old_label else None
        moved = label_moved = False
        if target != old_path and target.exists():
            raise ValueError(f"模型目录已存在: {target}")
        if old_label and new_label != old_label and new_label.exists():
            raise ValueError(f"标签映射文件已存在: {new_label}")
        try:
            if target != old_path and old_path.exists():
                old_path.rename(target)
                moved = True
            if old_label and new_label != old_label:
                old_label.rename(new_label)
                label_moved = True
            yield target
        except Exception:
            if label_moved:
                new_label.rename(old_label)
            if moved:
                target.rename(old_path)
            raise

    @staticmethod
    def _decode(row):
        """将数据库行转为字典，解码 JSON 及软删除布尔值，并添加类型和用途展示标签。

        Args:
            row: SQLite 查询得到的模型记录行。

        Returns:
            前端可用的模型详情字典。
        """
        result = dict(row)
        for key in ("params", "test_result"):
            result[key] = json.loads(result[key])
        result["is_deleted"] = bool(result["is_deleted"])
        result["model_label"] = MODEL_TYPE_LABELS.get(result["model"], result["model"])
        result["purpose_label"] = PURPOSE_EN_TO_ZH.get(result["purpose"], result["purpose"])
        return result

    def list(self, page=1, page_size=10):
        """按创建时间及 id 倒序分页读取未删除模型，裁剪完整报告为 test 指标概要。

        Args:
            page: 从 1 开始的页码。
            page_size: 每页记录数，允许 1 至 100，默认 10。

        Returns:
            包含 items、page、page_size、total、has_more 的字典；非法分页抛出 ValueError。

        Raises:
            ValueError: 页码小于 1 或每页数量不在 1 至 100 之间。
        """
        with _log_call("model_management.store.ModelStore.list", locals()):
            if page < 1 or not 1 <= page_size <= 100:
                raise ValueError("page 必须大于 0，page_size 必须在 1 到 100 之间")
            offset = (page - 1) * page_size
            with self._connect() as connection:
                total = connection.execute("SELECT COUNT(*) FROM models WHERE is_deleted=0").fetchone()[0]
                rows = connection.execute(
                    """SELECT * FROM models WHERE is_deleted=0
                    ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?""", (page_size, offset)).fetchall()
            # 列表只返回概要，完整分类报告等通过详情接口按需读取。
            items = []
            for row in rows:
                item = self._decode(row)
                evaluation = item.pop("test_result")
                item.pop("params")
                item["test_result"] = {"scores": evaluation.get("scores", {}), "rows": evaluation.get("rows")}
                items.append(item)
            return {"items": items, "page": page, "page_size": page_size,
                    "total": total, "has_more": offset + len(items) < total}

    def get(self, model_id):
        """按 id 读取未删除模型，返回完整参数和评估报告。

        Args:
            model_id: 模型登记记录的唯一 id。

        Returns:
            模型详情字典；不存在或已删除时抛出 KeyError。

        Raises:
            KeyError: 模型记录不存在或已经软删除。
        """
        with _log_call("model_management.store.ModelStore.get", locals()):
            with self._connect() as connection:
                row = connection.execute("SELECT * FROM models WHERE id=? AND is_deleted=0", (model_id,)).fetchone()
            if row is None:
                raise KeyError(model_id)
            return self._decode(row)

    def update(self, model_id, changes):
        """修改未删除模型，名称变更时同步重命名目录并更新路径。

        Args:
            model_id: 模型登记记录的唯一 id。
            changes: 待修改的 name 和/或 purpose 字段。

        Returns:
            修改后的详情字典；非法修改抛出 ValueError，记录不可用抛出 KeyError。

        Raises:
            ValueError: 修改字段不允许、字段值为空或超过长度限制。
            KeyError: 模型记录不存在或已经软删除。
        """
        with _log_call("model_management.store.ModelStore.update", locals()):
            if not changes or set(changes) - {"name", "purpose"}:
                raise ValueError("仅允许修改 name 和 purpose，且至少提供一个字段")
            if any(not isinstance(value, str) or not value.strip() or len(value.strip()) > 200
                   for value in changes.values()):
                raise ValueError("名称和用途必须为 1 到 200 个字符的非空字符串")
            record = self.get(model_id)
            changes = {key: value.strip() for key, value in changes.items()}
            with self._rename_directory(record["path"], changes.get("name", Path(record["path"]).name), record["label_path"]) as target:
                changes.update(name=target.name, path=str(target))
                if record["label_path"]:
                    changes["label_path"] = str(Path(record["label_path"]).with_name(f"{target.name}.json"))
                with self._connect() as connection:
                    cursor = connection.execute(
                        "UPDATE models SET " + ", ".join(f"{key}=?" for key in changes) +
                        " WHERE id=? AND is_deleted=0",
                        list(changes.values()) + [model_id])
                    if not cursor.rowcount:
                        raise KeyError(model_id)
            self._refresh_cache()
            return self.get(model_id)

    def list_ids(self):
        """返回当前未软删除模型的 ID 快照，用于跨页的全部删除。"""
        with self._connect() as connection:
            return [row["id"] for row in connection.execute(
                "SELECT id FROM models WHERE is_deleted=0 ORDER BY created_at DESC,id DESC")]

    def delete(self, model_id):
        """软删除模型，目录及数据库名称同步添加 _del_ 前缀。

        Args:
            model_id: 模型登记记录的唯一 id。

        Returns:
            包含 id 和 is_deleted=true 的字典；重复调用不重复加前缀，不存在时抛出 KeyError。

        Raises:
            KeyError: 模型记录不存在。
        """
        with _log_call("model_management.store.ModelStore.delete", locals()):
            with self._connect() as connection:
                row = connection.execute("SELECT * FROM models WHERE id=?", (model_id,)).fetchone()
            if row is None:
                raise KeyError(model_id)
            name = Path(row["path"]).name
            if not name.startswith("_del_"):
                name = "_del_" + name
            with self._rename_directory(row["path"], name, row["label_path"]) as target:
                with self._connect() as connection:
                    connection.execute("UPDATE models SET name=?, path=?, label_path=?, is_deleted=1 WHERE id=?",
                                       (target.name, str(target),
                                        str(Path(row["label_path"]).with_name(f"{target.name}.json")) if row["label_path"] else None,
                                        model_id))
            self._refresh_cache()
            return {"id": model_id, "is_deleted": True}

    def _refresh_cache(self):
        from inference.cache import catalog
        if self.path.resolve() == catalog.store.path.resolve():
            catalog.refresh()

    def inference_catalog(self):
        """读取全部可用模型，旧记录的标签文件迁移到数据库。"""
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM models WHERE is_deleted=0 ORDER BY created_at DESC,id DESC").fetchall()
            items = []
            for row in rows:
                labels = json.loads(row["labels"]) if row["labels"] else None
                if labels is None and row["label_path"]:
                    labels = json.loads(Path(row["label_path"]).read_text(encoding="utf-8"))
                    connection.execute("UPDATE models SET labels=? WHERE id=?", (json.dumps(labels, ensure_ascii=False), row["id"]))
                items.append({**{key: row[key] for key in ("id", "name", "model", "purpose", "created_at", "path", "params")}, "labels": labels or {}})
            return items
