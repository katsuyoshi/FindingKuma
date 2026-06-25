require "httparty"
require "json"
require "fileutils"

module FindingKuma
  class ImageFetcher
    BASE_URL = "https://kasenimg.pref.akita.lg.jp/cameraDataWeb/itv"
    SITES_JSON_URL = "https://kasen.pref.akita.lg.jp/pc/data/itv.json"
    IMAGE_DIR = File.expand_path("../../images", __dir__)

    def initialize
      FileUtils.mkdir_p(IMAGE_DIR)
    end

    def fetch_sites
      response = HTTParty.get(SITES_JSON_URL)
      body = response.body.force_encoding("Shift_JIS").encode("UTF-8")
      data = JSON.parse(body)
      data.delete("date")

      data.map do |id, info|
        {
          id: id.rjust(3, "0"),
          name: info["an"],
          city: info["ad"],
          river: info["rn"],
          lat: info["lat"],
          lng: info["lng"],
          status: info["flagStr"],
        }
      end.sort_by { |s| s[:id] }
    end

    def image_url(site_id, time)
      sn = site_id.to_s.rjust(3, "0")
      date_str = time.strftime("%Y%m%d")
      time_str = time.strftime("%Y%m%d%H%M")
      "#{BASE_URL}/#{date_str}/#{sn}/image_#{sn}_#{time_str}.jpg"
    end

    def fetch_image(site_id, time)
      url = image_url(site_id, time)
      sn = site_id.to_s.rjust(3, "0")
      time_str = time.strftime("%Y%m%d%H%M")
      filename = "#{sn}_#{time_str}.jpg"
      filepath = File.join(IMAGE_DIR, filename)

      response = HTTParty.get(url)
      if response.code == 200 && response.headers["content-type"]&.include?("image")
        File.binwrite(filepath, response.body)
        filepath
      else
        nil
      end
    end

    def fetch_recent(site_id, hours: 1)
      now = round_to_5min(Time.now)
      count = (hours * 60 / 5).to_i
      paths = []

      count.times do |i|
        time = now - (i * 5 * 60)
        path = fetch_image(site_id, time)
        paths << { time: time, path: path } if path
      end

      paths
    end

    def latest_image_url(site_id)
      sn = site_id.to_s
      "https://kasenimg.pref.akita.lg.jp/pc/img/camera_site/camera_site_#{sn}.jpg"
    end

    private

    def round_to_5min(time)
      min = time.min
      rounded_min = (min / 5) * 5
      Time.new(time.year, time.month, time.day, time.hour, rounded_min, 0)
    end
  end
end
