using Microsoft.ML;

namespace yuisanae2f.StrToStrAICS
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
        /// <returns>Model for the prediction</returns>
        protected ITransformer getModel(IDataView trainingDataView, IEstimator<ITransformer> pipeline)
        {
            var trainingPipeline = pipeline.Append(_mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features"))
                .Append(_mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));
            var _trainedModel = trainingPipeline.Fit(trainingDataView);
            return _trainedModel;
        }

        /// <param name="model">Model for the prediction</param>
        /// <param name="target">
        /// Input: <br/>
        /// make sure that you don't fill the output member in that context.
        /// </param>
        /// <returns>Would return a prediction for the <paramref name="target"/></returns>
        protected TPredict getPredict(PredictionEngine<T, TPredict> engine, T target)
        {
            return engine.Predict(target);
        }

        /// <param name="resArr">Parsed list of <typeparamref name="T"/> for the training.</param>
        /// <returns>DataView splited for the training.</returns>
        protected IDataView? splitDataView(T[] resArr)
        {
            return _mlContext.Data.LoadFromEnumerable(resArr);
        }

        /// <summary>
        /// Would save a model in given path as a file.
        /// </summary>
        /// <param name="path">Given model will be saved here</param>
        /// <param name="model">AI Model made</param>
        /// <param name="dataView">DataView splited. Check <c>splitDataView</c> method</param>
        protected void saveModel(string path, ITransformer model, IDataView dataView)
        {
            using (var fs = new FileStream(path, FileMode.Create))
                _mlContext.Model.Save(model, dataView.Schema, fs);
        }

        /// <param name="path">The path where model has been saved.</param>
        /// <returns>Loaded AI model from the file</returns>
        protected ITransformer loadModel(string path)
        {
            ITransformer model;

            using (var stream = new FileStream(path, FileMode.Open))
                model = _mlContext.Model.Load(stream, out var modelSchema);
            return model;
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
        /// Would save a model into zip file.
        /// </summary>
        /// <param name="path"></param>
        public void save(string path = "model.zip")
        {
            saveModel(path, model, dataView);
        }

        /// <summary>
        /// Would load a model from the zip file.
        /// </summary>
        /// <param name="path"></param>
        public void load(string path = "model.zip")
        {
            model = loadModel(path);
            engine = _mlContext.Model.CreatePredictionEngine<T, TPredict>(model);
            return;
        }

        /// <summary>
        /// Engine, will actually do predict.
        /// </summary>
        public PredictionEngine<T, TPredict> engine;

        /// <summary>
        /// Splited dataview
        /// </summary>
        public IDataView? dataView { get { return _dataView; } }
        private IDataView? _dataView;

        /// <summary>
        /// It would split this AI an dataView.
        /// </summary>
        /// <param name="resArr"></param>
        public void setDataView(T[] resArr) { _dataView = splitDataView(resArr); }

        /// <summary>
        /// AI Model, it would be the object for get result.
        /// </summary>
        protected ITransformer model;

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
            if (dataView == null) model = getModel(_dataView, pipeline);
            else if (_dataView == null) return;
            else model = getModel(dataView, pipeline);

            engine = _mlContext.Model.CreatePredictionEngine<T, TPredict>(model);
        }

        /// <summary>
        /// would run the model.
        /// </summary>
        /// <param name="target">Input Object. Make sure it is made of <typeparamref name="T"/></param>
        /// <returns>Predicted Value made of <typeparamref name="TPredict"/></returns>
        public TPredict predict(T target)
        {
            return getPredict(engine, target);
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