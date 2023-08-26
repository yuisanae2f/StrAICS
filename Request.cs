using Microsoft.ML.Data;

namespace yuisanae2f.StrAICS
{
    /// <summary>
    /// 
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public class Request<T>
    {
        [LoadColumn(0)]
        public string? input;

        [LoadColumn(1), ColumnName("Label")]
        public T? output;
    }
}