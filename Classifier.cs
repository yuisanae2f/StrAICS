using Microsoft.ML;

namespace yuisanae2f.StrAICS.ML
{
    /// <summary>
    /// StrToStr AI 101
    /// </summary>
    public class Classifier<T> : Root<T>
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
                _mlContext.Transforms.Conversion.MapValueToKey(inputColumnName: "output", outputColumnName: "Label")
                .Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: "input", outputColumnName: "inputFeaturised"))
                .Append(_mlContext.Transforms.Concatenate("Features", "inputFeaturised"))
                .Append(_mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features"))
                .Append(_mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));
        }
    }
}