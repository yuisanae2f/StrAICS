using Microsoft.ML.Data;

namespace yuisanae2f.StrMLCS
{
    /// <summary>
    /// An Input DataBase Format for this Context. <br/>
    /// In this context it would get string as the input, and set the output as a string also.
    /// </summary>
    public class Request<T>
    {
        [LoadColumn(0)]
        public string? input;

        [LoadColumn(1)]
        public T? output;
    }

    /// <summary>
    /// An Output DataBase Format for this Context. <br/>
    /// In this context it would get string as the input, and set the output as a string also.
    /// </summary>
    public class Response<T>
    {
        [ColumnName("PredictedLabel")]
        public T? predicted;

        [ColumnName("scores")]
        public float[] scores { get; set; }
    }
}