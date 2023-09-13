using Microsoft.ML;

namespace yuisanae2f.StrAICS.ML
{
    /// <summary>
    /// StrToStr AI 101
    /// </summary>
    public class Classifier<T> : Root<T, Request<T>, Response<T>>
    {
        /// <summary>
        /// Initialiser for this Object.
        /// </summary>
        /// <param name="mLContext">
        /// Custom mlContext. <br/>
        /// If not given it would generate by itself.
        /// </param>
        public Classifier(MLContext? mLContext = null) : base(mLContext) 
        {
            pipeline =
                ctx.Transforms.Conversion.MapValueToKey(inputColumnName: "Label", outputColumnName: "Label")
                .Append(ctx.Transforms.Text.FeaturizeText(inputColumnName: "input", outputColumnName: "inputFeaturised"))
                .Append(ctx.Transforms.Concatenate("Features", "inputFeaturised"))
                .Append(ctx.MulticlassClassification.Trainers.SdcaNonCalibrated("Label", "Features"))
                .Append(ctx.Transforms.Conversion.MapKeyToValue("PredictedLabel"));
        }

        public Response<T> predict(string target) { return getPredict(engine, new Request<T>() { input = target }); }
    }
}

// © 2023. YuiSanae2f