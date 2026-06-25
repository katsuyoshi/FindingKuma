require "json"
require "open3"

module FindingKuma
  class Detector
    SCRIPT_PATH = File.expand_path("../../scripts/detect.py", __dir__)

    def initialize(confidence: 0.3, save_img: false, results_dir: nil, classes: "all")
      @confidence = confidence
      @save_img = save_img
      @results_dir = results_dir || File.expand_path("../../results", __dir__)
      @classes = classes
    end

    def detect(image_path, baseline: nil)
      cmd = ["python3", SCRIPT_PATH, image_path,
             "--confidence", @confidence.to_s,
             "--classes", @classes]
      cmd += ["--baseline", baseline] if baseline
      cmd += ["--save-img", "--results-dir", @results_dir] if @save_img

      stdout, stderr, status = Open3.capture3(*cmd)

      unless status.success?
        return { "error" => "Detection failed: #{stderr.strip}", "image_path" => image_path }
      end

      json_line = stdout.lines.map(&:strip).reject(&:empty?).last
      JSON.parse(json_line)
    end

    def detected?(image_path, baseline: nil)
      result = detect(image_path, baseline: baseline)
      result.fetch("detection_count", 0) > 0
    end
  end
end
