"""maskveil — NSFW 部位検出と形状マスク生成の中核ロジック。

`core` は ComfyUI (`comfy`) に依存しない。依存は numpy と onnxruntime のみ。
パスの探索・検証は行わず、検証済みの絶対パスだけを受け取る。
"""
