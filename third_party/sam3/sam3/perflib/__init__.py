# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved


import os

is_enabled = False
if os.getenv("USE_PERFLIB", "1") == "1":

    is_enabled = True
