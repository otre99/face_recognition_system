#include "openvino_predictor.h"

shared_ptr<Predictor> OpenVinoPredictor::Create(const string &model_path,
                                    const string &config_path,
                                    const string &framework)
{
    auto impl = new OpenVinoPredictor();
    if (impl->Init(model_path, config_path, framework)){
        return shared_ptr<Predictor>(impl);
    } else {
        delete impl;
        return {};
    }
}


bool  OpenVinoPredictor::Init(const string &model_path, const string &config_path,
                             const string &framework){
    ov::Core core;
    std::shared_ptr<ov::Model> model = core.read_model(config_path, model_path);

    OPENVINO_ASSERT(model->inputs().size() == 1, "Sample supports models with 1 input only");

    cout << "OpenVinoPredictor Model name: " << model->get_friendly_name() << endl;
    const std::vector<ov::Output<ov::Node>> inputs = model->inputs();
    cout << "OpenVinoPredictor Inputs:" << endl;
    for (const ov::Output<ov::Node> &input : inputs) {
        const std::string name = input.get_names().empty() ? "NONE" : input.get_any_name();
        cout << "  Input name: " << name << endl;
        const ov::element::Type type = input.get_element_type();
        cout << "  Input type: " << type << endl;
        const ov::Shape shape = input.get_shape();
        cout << "  Input shape: " << shape << "\n\n";
    }
    cout << "OpenVinoPredictor Outputs:" << endl;
    const std::vector<ov::Output<ov::Node>> outputs = model->outputs();
    for (const ov::Output<ov::Node> &output : outputs) {
        const std::string name = output.get_names().empty() ? "NONE" : output.get_any_name();
        cout << "  Output name: " << name << endl;

        const ov::element::Type type = output.get_element_type();
        cout << "  Output type: " << type << endl;

        const ov::PartialShape shape = output.get_partial_shape();
        cout << "  Output shape: " << shape << "\n\n";
    }

    ov::preprocess::PrePostProcessor ppp(model);
    ppp.input().preprocess().resize(ov::preprocess::ResizeAlgorithm::RESIZE_LINEAR);
    ppp.input().model().set_layout("NCHW");
    ppp.output().tensor().set_element_type(ov::element::f32);

    model = ppp.build();



    /*
    // -------- Step 3. Set up input

    // Read input image to a tensor and set it to an infer request
    // without resize and layout conversions
    FormatReader::ReaderPtr reader(image_path.c_str());
    if (reader.get() == nullptr) {
        std::stringstream ss;
        ss << "Image " + image_path + " cannot be read!";
        throw std::logic_error(ss.str());
    }

    ov::element::Type input_type = ov::element::u8;
    ov::Shape input_shape = {1, reader->height(), reader->width(), 3};
    std::shared_ptr<unsigned char> input_data = reader->getData();

    // just wrap image data by ov::Tensor without allocating of new memory
    ov::Tensor input_tensor = ov::Tensor(input_type, input_shape, input_data.get());

    const ov::Layout tensor_layout{"NHWC"};

    // -------- Step 4. Configure preprocessing --------

    ov::preprocess::PrePostProcessor ppp(model);

    // 1) Set input tensor information:
    // - input() provides information about a single model input
    // - reuse precision and shape from already available `input_tensor`
    // - layout of data is 'NHWC'
    ppp.input().tensor().set_shape(input_shape).set_element_type(input_type).set_layout(tensor_layout);
    // 2) Adding explicit preprocessing steps:
    // - convert layout to 'NCHW' (from 'NHWC' specified above at tensor layout)
    // - apply linear resize from tensor spatial dims to model spatial dims
    ppp.input().preprocess().resize(ov::preprocess::ResizeAlgorithm::RESIZE_LINEAR);
    // 4) Here we suppose model has 'NCHW' layout for input
    ppp.input().model().set_layout("NCHW");
    // 5) Set output tensor information:
    // - precision of tensor is supposed to be 'f32'
    ppp.output().tensor().set_element_type(ov::element::f32);

    // 6) Apply preprocessing modifying the original 'model'
    model = ppp.build();

    // -------- Step 5. Loading a model to the device --------
    ov::CompiledModel compiled_model = core.compile_model(model, device_name);

    // -------- Step 6. Create an infer request --------
    ov::InferRequest infer_request = compiled_model.create_infer_request();
    // -----------------------------------------------------------------------------------------------------

    // -------- Step 7. Prepare input --------
    infer_request.set_input_tensor(input_tensor);

    // -------- Step 8. Do inference synchronously --------
    infer_request.infer();

    // -------- Step 9. Process output
    const ov::Tensor& output_tensor = infer_request.get_output_tensor();

    // Print classification results
    ClassificationResult classification_result(output_tensor, {image_path});
    classification_result.show();
    */
}

void OpenVinoPredictor::Predict(const cv::Mat &img, vector<cv::Mat> &outputs,
             const vector<string> &output_names) {


}

void  OpenVinoPredictor::Predict(const vector<cv::Mat> &images, vector<cv::Mat> &outputs,
                                const vector<string> &output_names) {

}
