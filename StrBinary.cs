using Microsoft.ML;
using Microsoft.ML.Data;

namespace yuisanae2f.StrAICS.ML
{
    namespace _Binary
    {
        public class Response : Response<bool>
        {
            /// <summary>
            /// The probability that given input could be predicted "true"
            /// </summary>
            [ColumnName("Probability")]
            public float probability { get; set; }

            /// <summary>
            /// The probability that the prediction had now isn't wrong
            /// </summary>
            public float score { get { return predicted ? probability : 1 - probability; } }
        }
    }

    public class Binary : Root<bool, _Binary.Response>
    {
        /// <summary>
        /// Initialiser for this Object.
        /// </summary>
        /// <param name="mLContext">
        /// Custom mlContext. <br/>
        /// If not given it would generate by itself.
        /// </param>
        public Binary(MLContext? mLContext = null) : base(mLContext)
        {
            pipeline =
                _mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: "input")
                .Append(_mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(
                    labelColumnName: "Label",
                    featureColumnName: "Features"
                    ));
        }
    }
}