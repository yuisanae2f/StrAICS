using Microsoft.ML;

namespace yuisanae2f.StrAICS.ML.Binary
{
    public class Binary : Root<bool, Request<bool>, Response>
    {
        public Binary(MLContext? mLContext = null) : base(mLContext)
        {
            pipeline =
                _mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: "input")
                .Append(_mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(
                    labelColumnName: "Label",
                    featureColumnName: "Features"
                    ));
        }

        public Response predict(string target) { return getPredict(engine, new Request<bool> { input = target }); }
    }
}