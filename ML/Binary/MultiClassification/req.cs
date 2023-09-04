using Microsoft.ML.Data;

namespace yuisanae2f.StrAICS.ML.Binary.MultiClassification
{
    public class req
    {
        [LoadColumn(0)] public string input;
        [LoadColumn(1)] public int cond;
        [LoadColumn(2), ColumnName("Label")] public bool output;
    }
}

// © 2023. YuiSanae2f