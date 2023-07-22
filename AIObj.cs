using Microsoft.ML;

namespace yuisanae2f.StrToStrAI
{
    /// <summary>
    /// FunctionBundle Root for the AIObj.<br/>
    /// Make an inheritance of this to make a new StrToStr thing.
    /// </summary>
    /// <typeparam name="T">Input Format for the Model (for the training also)</typeparam>
    /// <typeparam name="TPredict">Output Format for the Model</typeparam>
    public class _AIObj<T, TPredict>
        where T : class
        where TPredict : class, new()
    {
        /// <summary>
        /// Context for the each Machine Learning
        /// </summary>
        protected MLContext _mlContext;

        /// <summary>
        /// This context is specified for single str input and single str output.
        /// </summary>
        /// <returns>Fully processed pipeline for Learning</returns>
        protected IEstimator<ITransformer> processData()
        {
            var pipeline = _mlContext.Transforms.Conversion.MapValueToKey(inputColumnName: "output", outputColumnName: "Label");

            var pipeline2 = pipeline
                .Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: "input", outputColumnName: "inputFeaturised"));

            var pipeline3 = pipeline2
                .Append(_mlContext.Transforms.Concatenate("Features", "inputFeaturised"))
                .AppendCacheCheckpoint(_mlContext);

            return pipeline3;
        }

        /// <param name="trainingDataView">Fully processed DataView</param>
        /// <param name="pipeline">Fully processed pipeline for Learning</param>
        /// <returns>Engine for the prediction</returns>
        protected PredictionEngine<T, TPredict> getEngine(IDataView trainingDataView, IEstimator<ITransformer> pipeline)
        {
            var trainingPipeline = pipeline.Append(_mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features"))
                .Append(_mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));
            var _trainedModel = trainingPipeline.Fit(trainingDataView);
            var _predEngine = _mlContext.Model.CreatePredictionEngine<T, TPredict>(_trainedModel);
            return _predEngine;
        }

        /// <param name="model">Engine for the prediction</param>
        /// <param name="target">
        /// Input: <br/>
        /// make sure that you don't fill the output member in that context.
        /// </param>
        /// <returns>Would return a prediction for the <paramref name="target"/></returns>
        protected TPredict getPredict(PredictionEngine<T, TPredict> model, T target)
        {
            return model.Predict(target);
        }

        /// <param name="resArr">Parsed list of <typeparamref name="T"/> for the training.</param>
        /// <returns>DataView splited for the training.</returns>
        protected IDataView? splitDataView(T[] resArr)
        {
            return _mlContext.Data.LoadFromEnumerable(resArr);
        }
    }

    /// <summary>
    /// StrToStr AI 101
    /// </summary>
    /// <typeparam name="T">Input Format for the Model (for the training also)</typeparam>
    /// <typeparam name="TPredict">Output Format for the Model</typeparam>
    public class AIObj<T, TPredict> : _AIObj<T, TPredict>
        where T : class
        where TPredict : class, new()
    {
        /// <summary>
        /// Splited dataview
        /// </summary>
        public IDataView? dataView { get { return _dataView; } } private IDataView? _dataView;

        /// <summary>
        /// It would split this AI an dataView.
        /// </summary>
        /// <param name="resArr"></param>
        public void setDataView(T[] resArr) { _dataView = splitDataView(resArr); }

        /// <summary>
        /// Engine, it would be the object for get result.
        /// </summary>
        protected PredictionEngine<T, TPredict> engine;

        /// <summary>
        /// Pipeline for the engine training
        /// </summary>
        protected IEstimator<ITransformer> pipeline;

        /// <summary>
        /// Train the model with given dataView
        /// </summary>
        /// <param name="dataView">
        /// Set this param provided you want to train a model with specified dataView. <br/>
        /// Setting it null would consider that this model will be trained with given dataView(previous set).
        /// </param>
        public void train(IDataView? dataView = null)
        {
            if (dataView == null) engine = getEngine(_dataView, pipeline);
            else if (_dataView == null) return;
            else engine = getEngine(dataView, pipeline);
        }

        /// <summary>
        /// would run the model.
        /// </summary>
        /// <param name="target">Input Object. Make sure it is made of <typeparamref name="T"/></param>
        /// <returns>Predicted Value made of <typeparamref name="TPredict"/></returns>
        public TPredict predict(T target) {
            return engine.Predict(target);
        }

        /// <summary>
        /// Initialiser for this Object.
        /// </summary>
        /// <param name="mLContext">
        /// Custom mlContext. <br/>
        /// If not given it would generate by itself.
        /// </param>
        public AIObj(MLContext? mLContext = null)
        {
            if (mLContext == null) _mlContext = new MLContext(); else _mlContext = mLContext;
            pipeline = processData();
        }
    }
}