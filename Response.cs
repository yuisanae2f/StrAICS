using Microsoft.ML.Data;

namespace yuisanae2f.StrAICS
{
    public class Response<T>
    {
        [ColumnName("PredictedLabel")]
        public T? predicted { get; set; }
    }
}