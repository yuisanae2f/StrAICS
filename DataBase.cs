using Microsoft.ML.Data;

namespace yuisanae2f.StrToStrAI
{
    /// <summary>
    /// An Input DataBase Format for this Context. <br/>
    /// In this context it would get string as the input, and set the output as a string also.
    /// </summary>
    public class Request
    {
        [LoadColumn(0)]
        public string? input;

        [LoadColumn(1)]
        public string? output;
    }

    /// <summary>
    /// An Output DataBase Format for this Context. <br/>
    /// In this context it would get string as the input, and set the output as a string also.
    /// </summary>
    public class Response
    {
        [ColumnName("PredictedLabel")]
        public string? output;
    }

    public class Index
    {
        static int Main()
        {
            Console.WriteLine("StrToStrAI by LouiBooks");
            Console.WriteLine("Source: https://github.com/Louibooks/StrToStrAI");
            return 0;
        }
    }
}