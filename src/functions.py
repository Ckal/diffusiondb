from pandas import DataFrame

from zeno import (
    DistillReturn,
    MetricReturn,
    ModelReturn,
    ZenoOptions,
    distill,
    metric,
    model,
)


@model
def model_ret(name):
    def model(df: DataFrame, ops: ZenoOptions):
        return ModelReturn(model_output=df[ops.data_column])

    return model


@distill
def length(df: DataFrame, ops: ZenoOptions):
    return DistillReturn(distill_output=df["prompt"].str.len())


@metric
def avg_image_nswf(df: DataFrame, ops: ZenoOptions):
    return MetricReturn(metric=float(df["image_nsfw"].dropna().mean()))


@metric
def avg_prompt_nsfw(df: DataFrame, ops: ZenoOptions):
    return MetricReturn(metric=float(df["prompt_nsfw"].dropna().mean()))
