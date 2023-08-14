using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Linq;
using Tensorflow.Contexts;
using TensorFlow;

namespace yuisanae2f.StrAICS.ML
{
    namespace _Binary
    {
        public class Response
        {
            [ColumnName("PredictedLabel")]
            bool predicted { get; set; }
            [ColumnName("Probability")]
            public float probability { get; set; }

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