using Microsoft.ML.Data;

namespace yuisanae2f.StrAICS.ML.Binary
{
    public class Response : Response<bool>
    {
        [ColumnName("Probability")]
        public float probability { get; set; }

        public float score { get { return predicted ? probability : 1 - probability; } }
    }
}
