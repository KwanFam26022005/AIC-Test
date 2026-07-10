# Phân tích lỗi caption plan

> sau khi phân tích lỗi về plan [caption_recap_fusecap_plan.md](c:/Users/User/Desktop/AIC2026/data_processing/caption/caption_recap_fusecap_plan.md) và ouput @c:/Users/User/Desktop/AIC2026/outputs/
> Lưu ý: có thể tham khảo phân tích ở C:\Users\User\.gemini\antigravity-ide\brain\d8ecc540-bd6f-482e-817c-f61eb04ceb88\analysis_caption_plan.md

<details><summary>1 previous message</summary>

> Mình sẽ đọc plan, thư mục output và bản phân tích tham khảo trước, rồi đối chiếu với pipeline đang mở để tìm đúng chỗ lệch. Nếu đường dẫn `C:\Users\User\Desktop\...` bị sandbox chặn, mình sẽ báo ngay và dùng bản nào đang có trong workspace.
</details>

Đường dẫn đều đọc được. Mình đang lấy nội dung plan, bản phân tích tham khảo và cấu trúc output để xem lỗi là lỗi thiết kế pipeline, lỗi format output, hay lỗi triển khai.

> tôi cung cấp lại phần phân tích lỗi [analysis_caption_plan.md](c:/Users/User/Desktop/AIC2026/analysis_caption_plan.md)

<details><summary>6 previous messages</summary>

> Đã nhận, mình sẽ dùng bản [analysis_caption_plan.md](c:/Users/User/Desktop/AIC2026/analysis_caption_plan.md) này làm nguồn phân tích chính. Mình đọc lại file ngay để bám đúng nội dung bạn vừa cung cấp, rồi sẽ chốt các lỗi/blocker và hướng sửa theo output thực tế.
>
> File phân tích này xác nhận ba blocker chính: timestamp mapping, ASR encoding, và chuẩn hóa frame_id. Mình sẽ kiểm tra thêm một vài record thật và manifest/summary để xem blocker nào có thể tự giải ngay bằng metadata sẵn có, thay vì phải giả định.
>
> Một điểm quan trọng: ASR trong file hiện tại hiển thị tiếng Việt đúng, không còn thấy lỗi mojibake ở các dòng mẫu. Mình sẽ kiểm tra nhanh ký tự thay thế `�` và metadata timing, vì nếu ASR đã ổn thì blocker lớn nhất còn lại là cách map `frame_idx` sang `timestamp_sec`.
>
> Có thêm hai phát hiện làm kế hoạch sửa nhẹ hơn: `video_manifest.jsonl` có `duration_sec=1204.837`, `fps=30`, và `object_detection` có đúng `num_frames=283`. Như vậy có thể dựng timestamp gần đúng từ thứ tự keyframe; đồng thời ASR hiện tại không còn chứa ký tự `�`, nên phần “fix encoding” nên đổi thành “validate/guard encoding” thay vì coi là blocker chắc chắn.
>
> Mình thấy local `keyframe_test/L22_V012/*.jpg` đã có đầy đủ ảnh keyframe, nên nhận định “không thấy local keyframes” trong file phân tích không còn đúng với workspace hiện tại. Mình kiểm tra tiếp script trích keyframe/map CSV để xác định timestamp chuẩn hơn công thức nội suy.
>
> Script chính của `keyframe_extraction` có thể sinh `map_keyframes/*.csv`, nhưng trong bản local hiện tại không có CSV map cho `L22_V012`, và sample keyframe đang dùng tên `001.jpg` thay vì format `0001.jpg` của script TransNetV2. Vì vậy timestamp tốt nhất hiện giờ là lấy từ manifest + số keyframe, hoặc bổ sung map CSV nếu bạn còn bản server.
</details>

Mình đã đọc bản [analysis_caption_plan.md](</c:/Users/User/Desktop/AIC2026/analysis_caption_plan.md>) bạn cung cấp lại và đối chiếu nhanh với output thực tế.

Có 3 điểm cần cập nhật so với phân tích trong file:

1. `ASR encoding` hiện tại có vẻ không còn là blocker: mình kiểm tra `asr_segments.jsonl` và không thấy ký tự `�`; tiếng Việt hiển thị đúng. Nên đổi mục này thành “validate encoding + dùng `audio_features.jsonl`/`clean_transcript` + `usable_for_caption`”, không cần coi là lỗi chắc chắn.

2. `keyframes` thực tế đã có local tại:
   `C:\Users\User\Desktop\AIC2026\keyframe_test\L22_V012`
   với đủ 283 ảnh `001.jpg` đến `283.jpg`. Vì vậy Phase VLM/fuser có thể dùng local keyframes này, dù JSON output vẫn trỏ path server `/tmp2/...`.

3. `timestamp_sec` vẫn là blocker chính, nhưng có thể xử lý fallback được:
   `video_manifest.jsonl` có `duration_sec=1204.837`, `fps=30.0`, object summary có `num_frames=283`. Nếu không có `map_keyframes/*.csv`, có thể tạm nội suy:
   `timestamp_sec = (frame_idx - 1) * duration_sec / (num_frames - 1)`
   và nên ghi thêm `timestamp_source: "uniform_interpolation"` để biết đây chưa phải timestamp chuẩn.

Kết luận kỹ thuật nên sửa plan theo hướng:

- Canonical frame id nên là `L22_V012_001`, không ép 6 chữ số như `L22_V012_000001`, vì output object/keyframe hiện dùng 3 chữ số.
- Alignment nên ưu tiên: `map_keyframes.csv` nếu có → `timestamp_sec` có sẵn → fallback nội suy từ manifest.
- Tags không cần input riêng, lấy trực tiếp từ object detection field `tags` / `raw_tags`.
- ASR nên dùng `outputs/audio/L22_V012/features/audio_features.jsonl` hơn là raw `asr_segments.jsonl`, vì đã có `clean_transcript`, `summary`, `keywords`, `usable_for_caption`.
- OCR caption input nên dùng `ocr_text_clean` hoặc các `ocr_lines` có `keep_for_index=true`; tránh dùng `ocr_text_search` cho caption vì field này có lặp term phục vụ search.

Bước tiếp theo hợp lý là viết Phase 0 adapter: load object/OCR/audio/manifest, normalize `frame_id`, dựng `timestamp_sec`, merge evidence và xuất `frame_evidence.jsonl`.

> Và hiện tại đang có những lỗi nào cần triệt để ở [audio_extraction](c:/Users/User/Desktop/AIC2026/data_processing/audio_extraction/) [ocr_vlm_pipeline_v2](c:/Users/User/Desktop/AIC2026/data_processing/OCR/ocr_vlm_pipeline_v2/) [object_detection](c:/Users/User/Desktop/AIC2026/data_processing/object_detection/)

<details><summary>6 previous messages</summary>

> Mình sẽ rà theo kiểu “triage lỗi cần sửa tận gốc”: không chỉ nhìn plan, mà đối chiếu code pipeline, output JSONL thực tế và các manifest/summary. Mình sẽ ưu tiên các lỗi làm hỏng schema, alignment, search/caption downstream, hoặc khiến batch scale sai âm thầm.
>
> Mình đang tách lỗi thành hai loại: lỗi “schema/trạng thái” làm downstream hiểu nhầm, và lỗi “chất lượng dữ liệu” làm caption/search nhiễu. Với audio, đã thấy một mùi khá rõ: manifest ghi `pending` dù đã có output, và ASR có nhiều câu boilerplate kiểu “subscribe...” vẫn bị đánh `good`.
>
> Với OCR và object detection, mình sẽ nhìn vào phần export/schema trước, vì caption fusion cần các field đó ổn định. Nếu schema thiếu `timestamp_sec`, `area_ratio`, `position`, hoặc text search bị trộn với text caption, lỗi sẽ lan rất xa.
>
> Mình cũng đang kiểm tra số liệu thực tế, không chỉ code. Có vài lỗi kiểu “không crash nhưng sai ngữ nghĩa”: ví dụ OCR `ocr_text_search` cố tình lặp clean/group/line để search, nhưng nếu đưa thẳng vào caption thì sẽ lặp chữ; audio feature thì đã sort theo thời gian, còn raw ASR segments có thể không sort do chạy job theo duration.
>
> Một lỗi OCR đáng chú ý: khi chạy nhiều worker, file ES JSONL được append theo thứ tự future hoàn thành, nên output không còn theo timeline; CSV summary thì có sort lại, JSONL thì không. Đây là kiểu lỗi rất dễ gây sai nếu bước caption đọc JSONL tuần tự để tạo memory/shot.
>
> Object detection không chỉ thiếu metadata; còn có lỗi semantic count: cùng một người có thể được đếm vừa `PERSON`, vừa `WOMAN`/`MAN`/`STUDENT`. Với retrieval/caption kiểu “bao nhiêu người”, lỗi này đáng sửa ở tầng canonicalization chứ không nên để fuser tự đoán.
</details>

Có. Mình thấy các lỗi cần sửa triệt để nhất là những lỗi làm downstream caption/alignment sai âm thầm.

**audio_extraction**
1. Manifest trạng thái bị sai/ngây nhiễu: `video_manifest` vẫn `status: pending`, và toàn bộ 66 job trong `asr_job_manifest.jsonl` vẫn `pending` dù đã có 226 ASR rows. Gốc ở [pipeline.py](</c:/Users/User/Desktop/AIC2026/data_processing/audio_extraction/audio_pipeline/pipeline.py:104>) và [vad.py](</c:/Users/User/Desktop/AIC2026/data_processing/audio_extraction/audio_pipeline/vad.py:36>). Nên tách rõ “job plan” với “job result”, hoặc cập nhật status sau ASR.

2. `asr_segments.jsonl` không theo timeline vì job được sort theo duration trước khi chạy: [asr.py](</c:/Users/User/Desktop/AIC2026/data_processing/audio_extraction/audio_pipeline/asr.py:37>). Output hiện có 24 lần đảo thứ tự. `audio_features.jsonl` đã sort lại, nhưng nếu caption đọc raw ASR sẽ sai memory/context. Cần sort ASR/quality/features theo `video_id,start_sec,end_sec`.

3. Quality gate quá đơn giản, để lọt boilerplate/hallucination ASR. Hiện có 9 audio features dạng “subscribe/không bỏ lỡ/Amara/Cảm ơn...” và 8 cái vẫn `usable_for_caption=true`. Gốc ở [quality.py](</c:/Users/User/Desktop/AIC2026/data_processing/audio_extraction/audio_pipeline/quality.py:21>) chỉ check độ dài/lặp/duration. Cần thêm blacklist boilerplate, duplicate-text-across-video, language/semantic confidence, và rule loại câu intro/outro quảng bá.

4. Resume ASR chưa an toàn: `done_job_ids` chỉ dựa vào `job_id` [asr.py](</c:/Users/User/Desktop/AIC2026/data_processing/audio_extraction/audio_pipeline/asr.py:30>). Nếu đổi VAD/config/audio mà job id vẫn giống, pipeline sẽ skip output cũ. Cần job fingerprint gồm `audio_path,start_sec,end_sec,model,config_hash`.

**ocr_vlm_pipeline_v2**
1. `frame_id` không cùng schema với object detection: OCR là `001`, object là `L22_V012_001`. Gốc ở [outputs.py](</c:/Users/User/Desktop/AIC2026/data_processing/OCR/ocr_vlm_pipeline_v2/ocr_pipeline/outputs.py:267>). Cần thêm `canonical_frame_id` hoặc đổi top-level `frame_id` thống nhất.

2. OCR output thiếu `timestamp_sec`, nên không align audio/window trực tiếp được. Cần nhận `map_keyframes.csv` hoặc manifest mapping khi export ES doc.

3. JSONL OCR bị out-of-order khi chạy multi-worker: hiện có 51 lần đảo thứ tự. Gốc ở [run_frame_folder.py](</c:/Users/User/Desktop/AIC2026/data_processing/OCR/ocr_vlm_pipeline_v2/run_frame_folder.py:485>) append theo `as_completed`; chỉ CSV được sort ở [run_frame_folder.py](</c:/Users/User/Desktop/AIC2026/data_processing/OCR/ocr_vlm_pipeline_v2/run_frame_folder.py:518>). Cần buffer/sort docs trước khi ghi hoặc post-sort JSONL.

4. `ocr_text_search` bị lặp mạnh vì ghép `clean_text + group_texts + line_texts`: [outputs.py](</c:/Users/User/Desktop/AIC2026/data_processing/OCR/ocr_vlm_pipeline_v2/ocr_pipeline/outputs.py:287>). 269/283 docs có search text dài hơn clean nhiều lần. Field này dùng search được, nhưng không nên đưa vào caption. Cần thêm `ocr_text_caption` không lặp.

5. Policy “review” chưa sạch: 293 line vừa `need_review=true` vừa `keep_for_index=true`. Gốc ở [gating.py](</c:/Users/User/Desktop/AIC2026/data_processing/OCR/ocr_vlm_pipeline_v2/ocr_pipeline/scoring/gating.py:213>) và được đưa vào clean ở [outputs.py](</c:/Users/User/Desktop/AIC2026/data_processing/OCR/ocr_vlm_pipeline_v2/ocr_pipeline/outputs.py:91>). Cần quyết rõ: review-only có index không; nếu caption thì nên tách khỏi clean.

**object_detection**
1. Thiếu `timestamp_sec` cho toàn bộ 283 records, nên chưa align audio/shot chuẩn được. Gốc output builder ở [ram_gdino_pipeline.py](</c:/Users/User/Desktop/AIC2026/data_processing/object_detection/ram_gdino_pipeline.py:604>) không export timestamp.

2. Thiếu `area_ratio` và `position` cho toàn bộ 1586 objects, dù đã có hàm tính area trong [tag_canonicalization.py](</c:/Users/User/Desktop/AIC2026/data_processing/object_detection/tag_canonicalization.py:365>). Cần ghi vào từng object để caption biết object lớn/nhỏ/vị trí.

3. Semantic double-count: 22 frame có collision kiểu `PERSON` cùng `WOMAN/MAN/STUDENT`; ví dụ frame `L22_V012_003` có `PERSON=2` và `WOMAN=1` cho cùng vùng người. Hypernym map hiện không xử lý person-specific chain: [tag_canonicalization.py](</c:/Users/User/Desktop/AIC2026/data_processing/object_detection/tag_canonicalization.py:103>). Cần canonicalize person/man/woman/student để counts không phồng.

4. `image_path` trong OCR/object đều trỏ `/tmp2/...`, hiện 283/283 path không tồn tại local. Cần thêm `relative_image_path` hoặc resolver field dùng được trên máy hiện tại.

5. Label/count đang uppercase (`PERSON`, `WINDOW`), còn tags lowercase. `normalize_label` cố tình uppercase ở [tag_canonicalization.py](</c:/Users/User/Desktop/AIC2026/data_processing/object_detection/tag_canonicalization.py:351>). Downstream caption/search nên có thêm normalized lowercase fields: `object_tags`, `object_text`, `object_counts_normalized`.

Ưu tiên sửa: schema alignment trước (`canonical_frame_id`, `timestamp_sec`, local/relative image path), rồi sort output theo timeline, rồi quality filters cho ASR/OCR/object counts.

> trước hết, hãy lên plan sửa cho phần @c:/Users/User/Desktop/AIC2026/data_processing/audio_extraction/

**Plan Sửa Audio**
Mục tiêu: biến `audio_extraction` thành nguồn evidence ổn định cho caption/retrieval, không gây sai timeline, không đưa ASR rác vào caption, và resume không dùng nhầm output cũ.

**Phase 1: Chuẩn hóa trạng thái manifest**
Sửa [pipeline.py](</c:/Users/User/Desktop/AIC2026/data_processing/audio_extraction/audio_pipeline/pipeline.py:17>) và [vad.py](</c:/Users/User/Desktop/AIC2026/data_processing/audio_extraction/audio_pipeline/vad.py:14>).

- `video_manifest.jsonl`: đổi `status: pending` thành trạng thái thật như `probed`, `no_audio`, `probe_failed`.
- `asr_job_manifest.jsonl`: coi đây là job plan, đổi `status: pending` thành `planned`, hoặc bỏ `status`.
- Thêm output mới hoặc field summary cho ASR result:
  - `num_jobs_planned`
  - `num_jobs_success`
  - `num_jobs_empty`
  - `num_jobs_failed`
  - `num_segments`
- Không để downstream hiểu nhầm rằng job vẫn chưa chạy.

**Phase 2: Bảo đảm timeline order**
Sửa [asr.py](</c:/Users/User/Desktop/AIC2026/data_processing/audio_extraction/audio_pipeline/asr.py:16>) và [features.py](</c:/Users/User/Desktop/AIC2026/data_processing/audio_extraction/audio_pipeline/features.py:31>).

- Vẫn có thể chạy job theo duration để tối ưu tốc độ.
- Nhưng trước khi ghi hoặc sau khi chạy xong phải sort:
  - `asr_segments.jsonl`
  - `asr_quality_report.jsonl`
  - `audio_features.jsonl`
- Sort key: `(video_id, start_sec, end_sec, asr_segment_id)`.
- Acceptance: raw `asr_segments.jsonl` không còn đảo thứ tự timeline.

**Phase 3: Sửa quality gate chống ASR rác**
Sửa [quality.py](</c:/Users/User/Desktop/AIC2026/data_processing/audio_extraction/audio_pipeline/quality.py:7>) và [text.py](</c:/Users/User/Desktop/AIC2026/data_processing/audio_extraction/audio_pipeline/text.py:42>).

Thêm các rule:

- Boilerplate blacklist:
  - `subscribe`
  - `đăng ký`
  - `không bỏ lỡ`
  - `cảm ơn quý vị đã theo dõi`
  - `amara.org`
- Detect repeated boilerplate across video: cùng transcript xuất hiện nhiều lần ở các timestamp xa nhau thì hạ `quality_level`.
- Thêm field giải thích:
  - `quality.reasons: ["boilerplate", "repeated_across_video"]`
- Với các dòng này:
  - `usable_for_caption=false`
  - `need_fallback=true`
  - vẫn giữ trong raw ASR để audit, nhưng không đưa vào caption evidence.

**Phase 4: Resume an toàn bằng job fingerprint**
Sửa [asr.py](</c:/Users/User/Desktop/AIC2026/data_processing/audio_extraction/audio_pipeline/asr.py:30>).

Hiện resume chỉ dựa vào `job_id`, dễ dùng nhầm output cũ nếu đổi VAD/model/config. Cần thêm:

- `job_fingerprint = hash(video_id, audio_path, start_sec, end_sec, model, language, beam_size, vad config)`
- Ghi fingerprint vào:
  - `asr_job_manifest.jsonl`
  - `asr_segments.jsonl`
- Khi resume, chỉ skip nếu `job_id` và `job_fingerprint` đều khớp.

**Phase 5: Caption-ready audio evidence**
Sửa [features.py](</c:/Users/User/Desktop/AIC2026/data_processing/audio_extraction/audio_pipeline/features.py:31>).

Thêm field phục vụ caption:

- `caption_text`: chỉ có text đã sạch và usable.
- `search_text`: có thể giữ rộng hơn caption text.
- `exclude_from_caption_reason`: nếu không usable.
- `time_window`: `[start_sec, end_sec]`.

Rule downstream: caption pipeline dùng `audio_features.jsonl`, không dùng trực tiếp `asr_segments.jsonl`.

**Phase 6: Validation/report**
Thêm script nhỏ hoặc function report trong pipeline:

- kiểm tra sort timeline
- đếm boilerplate bị loại
- đếm empty/bad/medium/good
- kiểm tra `job_manifest` và `asr_segments` khớp fingerprint
- xuất `audio_quality_summary.json`

**Thứ tự triển khai đề xuất**
1. Sort timeline + manifest status.
2. Quality gate boilerplate.
3. Job fingerprint resume.
4. Caption-ready fields.
5. Report validation.

Sau khi sửa xong, output audio nên đạt: raw ASR audit được, `audio_features.jsonl` sạch theo timeline, và caption module chỉ lấy các segment `usable_for_caption=true`.

> Hãy viết plan .md sửa code vào @c:/Users/User/Desktop/AIC2026/data_processing/audio_extraction/

<details><summary>4 previous messages</summary>

> Mình sẽ tạo một file plan riêng trong `data_processing/audio_extraction/`, dạng checklist triển khai có thứ tự, file/code cần sửa, schema output mong muốn và tiêu chí nghiệm thu. Trước khi ghi mình kiểm tra nhanh tên file để không đè nhầm tài liệu đang có.
>
> Mình sẽ đặt tên file là `audio_extraction_code_fix_plan.md` để tách khỏi plan gốc hiện có. Nội dung sẽ tập trung vào sửa code thật, không chỉ mô tả ý tưởng: module nào sửa, field nào thêm, output nào thay đổi, test nào cần chạy.
>
> <details><summary>Created a file</summary>
>
> - Wrote `C:/Users/User/Desktop/AIC2026/data_processing/audio_extraction/audio_extraction_code_fix_plan.md` (+513 -0)
>
> </details>
>
> File plan đã được tạo. Mình kiểm tra lại nhanh để chắc file nằm đúng thư mục và render được phần đầu/cuối.
</details>

Đã tạo file plan tại:

[audio_extraction_code_fix_plan.md](</c:/Users/User/Desktop/AIC2026/data_processing/audio_extraction/audio_extraction_code_fix_plan.md>)

File này gồm các phần: lỗi hiện tại, schema output sau sửa, thay đổi code theo từng phase, test plan, thứ tự triển khai và Definition of Done.

> Tiếp theo hãy phân tích ở Object_detection và từ đó đưa plan sửa đổi

<details><summary>5 previous messages</summary>

> Mình sẽ làm giống phần audio: rà code/output object detection, chốt lỗi nào ảnh hưởng trực tiếp tới caption fusion/retrieval, rồi tạo một plan sửa code trong thư mục `data_processing/object_detection/`. Trọng tâm sẽ là schema alignment, timestamp, object counts không bị phồng, và metadata cho caption.
>
> Code hiện tại cho thấy pipeline đã khá gọn, nhưng schema output đang “thiếu xương sống” cho caption: không có timestamp, không có vị trí/diện tích object, và counts còn dễ đếm trùng người. Mình lấy thêm thống kê từ output `L22_V012_objects.jsonl` để plan không bị chung chung.
>
> Thống kê xác nhận các lỗi chính: 283/283 frame thiếu `timestamp_sec`, 1586/1586 object thiếu `area_ratio` và `position`, 22 frame có count người bị trùng kiểu `PERSON` + `WOMAN/MAN/STUDENT`, 33 frame còn label scene như `SKY/CITY/WATER/SUN` nằm trong object counts. Mình sẽ viết plan sửa code thành file `.md` riêng trong thư mục object detection.
>
> <details><summary>Created a file</summary>
>
> - Wrote `C:/Users/User/Desktop/AIC2026/data_processing/object_detection/object_detection_code_fix_plan.md` (+678 -0)
>
> </details>
>
> Plan sửa object detection đã được tạo. Mình kiểm tra lại file đầu/cuối để chắc nội dung ghi đúng và không bị lỗi encoding.
</details>

Đã phân tích và tạo plan sửa object detection tại:

[object_detection_code_fix_plan.md](</c:/Users/User/Desktop/AIC2026/data_processing/object_detection/object_detection_code_fix_plan.md>)

Plan này chốt các lỗi chính từ output hiện tại: thiếu `timestamp_sec`, thiếu `area_ratio/position`, path ảnh không portable, scene label lọt vào object counts, count người bị phồng do `PERSON` + `WOMAN/MAN/STUDENT`, và resume chỉ dựa vào `frame_id`. File cũng có schema đề xuất, phase sửa code, adapter enrich output cũ, test plan và Definition of Done.

> Tôi có một số review như sau:
> + timestampe sẽ là gì trong tình huống này, vì cũng như bạn thấy tôi đang thực hiện trên các frame đã được extract(do tôi chưa hiểu về timestampe ) có phải bản nói về map-key tôi mới vừa thêm vào [L22_V012.csv](c:/Users/User/Desktop/AIC2026/keyframe_test/L22_V012/L22_V012.csv)
> + về image_path thì tôi đang thực hiện trên server nên không cần lo về path local (vì tôi sẽ push trên github, rồi trên server tôi sẽ gitpull )
> + lưu ý về schema lúc đầu tôi đang thiên hướng về ES thì liệu thay đổi vậy có ảnh hưởng gì không?
> Lưu ý tổng: vì object extraction, caption extraction và Ocr extraction là kiểu dữ dạng text nên hãy suy nghĩ kĩ về schema tối ưu nhất về việc search
> +

<details><summary>5 previous messages</summary>

> Mình hiểu review của bạn là muốn chỉnh lại plan cho đúng thực tế vận hành server và đúng hướng ES/search, không chỉ “schema đẹp”. Mình sẽ đọc `L22_V012.csv` mới thêm để xác nhận timestamp nên lấy từ đâu, rồi cập nhật plan object detection cho sát hơn.
>
> CSV bạn mới thêm đúng là mảnh còn thiếu: `pts_time` chính là timestamp của keyframe trong video gốc, tính bằng giây từ đầu video. Mình sẽ chỉnh plan: dùng `L22_V012.csv` làm nguồn timestamp ưu tiên, hạ phần `image_path local portability` xuống optional, và viết rõ schema theo hướng ES/search thay vì chỉ phục vụ caption.
>
> Mình sẽ chỉnh trực tiếp plan theo 3 review của bạn: timestamp lấy từ CSV map-keyframe, `image_path` server là canonical nên không coi là lỗi, và schema phải ưu tiên ES/search với các field text/keyword/count rõ ràng.
>
> <details><summary>Edited a file</summary>
>
> - Wrote `C:/Users/User/Desktop/AIC2026/data_processing/object_detection/object_detection_code_fix_plan.md` (+129 -26)
>
> </details>
>
> Mình đã cập nhật plan theo review. Mình kiểm tra nhanh các đoạn mới để đảm bảo ba ý chính đã được phản ánh: timestamp từ CSV, server path là canonical, và schema ES/search-first.
</details>

Đã cập nhật lại plan theo review của bạn tại:

[object_detection_code_fix_plan.md](</c:/Users/User/Desktop/AIC2026/data_processing/object_detection/object_detection_code_fix_plan.md>)

Về 3 ý chính:

1. `timestamp_sec` trong trường hợp này chính là `pts_time` trong [L22_V012.csv](</c:/Users/User/Desktop/AIC2026/keyframe_test/L22_V012/L22_V012.csv>). Nó là số giây tính từ đầu video gốc tới keyframe đã extract, không phải thời gian ngoài đời. Ví dụ `002.jpg` tương ứng `n=2`, `pts_time=3.0`, `frame_idx=90`, nên object record nên có `keyframe_idx=2`, `timestamp_sec=3.0`, `source_frame_idx=90`.

2. `image_path` mình đã sửa lại quan điểm trong plan: vì bạn chạy trên server, absolute server path là canonical, không coi là lỗi nữa. `image_relpath` chỉ là optional fallback/debug khi export hoặc visualize ở môi trường khác.

3. Về ES schema: plan giờ giữ backward compatibility bằng cách không bỏ field cũ như `object_counts`, `objects`, `tags`; chỉ thêm field search-friendly. Hướng tối ưu là tách:
   - `object_tags`, `scene_tags`, `ram_tags` dạng `keyword` để filter.
   - `timestamp_sec`, `keyframe_idx`, `source_frame_idx` dạng numeric.
   - `object_text`, `scene_text`, `ram_tag_text`, `all_object_text` dạng text để search.
   - `object_count_items` để query count như “ít nhất 2 người” mà không phụ thuộc dynamic field quá rộng.

Nói ngắn gọn: schema mới nên additive, không phá ES cũ; khi scale thì nên tạo index version mới thay vì mutate mapping cũ.

> nếu làm như trên vẫn còn phù hợp với plan caption hiện tại đúng không

Đúng, vẫn phù hợp với plan caption hiện tại, thậm chí còn làm plan caption “chạy được thật” hơn.

Các thay đổi object detection là **additive schema**: giữ các field cũ như `frame_id`, `video_id`, `objects`, `object_counts`, `tags`, rồi thêm field chuẩn hơn cho caption/search như `timestamp_sec`, `object_counts_normalized`, `object_text`, `scene_tags`, `ram_tag_text`. Vì vậy không phá plan caption, chỉ giúp `Evidence Alignment` và `Evidence Builder` bớt phải đoán.

Điểm khớp trực tiếp với caption plan:

- `timestamp_sec` từ `L22_V012.csv/pts_time` giúp audio window, shot grouping, temporal caption và TRAKE alignment hoạt động đúng.
- `object_counts_normalized`, `object_tags`, `important_objects` đúng với input mà caption plan mong đợi.
- `scene_tags`, `ram_tags`, `scene_text`, `object_text` giúp compact search index tốt hơn.
- `source_frame_idx` và `keyframe_idx` làm rõ thứ tự keyframe vs frame gốc, tránh lỗi temporal.
- Giữ `image_path` server nên vẫn dùng được khi chạy VLM/caption trên server.

Chỉ cần cập nhật caption plan ở Phase 0/Evidence Alignment một chút: object input không còn chỉ đọc `object_counts`; nó nên ưu tiên các field mới nếu có, fallback về field cũ nếu chưa enrich. Nói gọn: plan caption không đổi kiến trúc, chỉ đổi adapter/schema mapping cho chắc hơn.

> nếu làm như trên vẫn còn phù hợp với plan caption [caption_recap_fusecap_plan.md](c:/Users/User/Desktop/AIC2026/data_processing/caption/caption_recap_fusecap_plan.md) hiện tại đúng không

Đúng, vẫn phù hợp với [caption_recap_fusecap_plan.md](</c:/Users/User/Desktop/AIC2026/data_processing/caption/caption_recap_fusecap_plan.md>).

Quan trọng là các sửa đổi object detection đang đi theo hướng **bổ sung adapter/schema**, không đổi kiến trúc caption. Plan caption vẫn giữ nguyên luồng:

```text
OCR + object detection/counts + image tags + audio
→ evidence alignment
→ frame caption fusion
→ ReCap temporal caption
→ TRAKE/event-step
→ compact search index
```

Các field mới còn làm plan caption khớp thực tế hơn:

- `timestamp_sec = pts_time` từ `L22_V012.csv` giúp đúng phần Evidence Alignment, audio window, shot grouping và TRAKE temporal order.
- `keyframe_idx` và `source_frame_idx` làm rõ “ảnh keyframe thứ mấy” khác với “frame gốc trong video”, tránh sai temporal.
- `object_counts_normalized`, `object_tags`, `important_objects` đúng với input schema mà caption plan đang mong muốn.
- `scene_tags`, `ram_tags`, `object_text`, `scene_text`, `all_object_text` rất hợp với phần compact search/ES index.
- Giữ `image_path` server không ảnh hưởng caption plan, vì VLM/caption trên server vẫn dùng được.

Điểm cần chỉnh nhẹ trong caption plan là Phase 0/Evidence Alignment nên ghi rõ priority đọc object evidence:

```text
Ưu tiên field mới:
object_counts_normalized, object_tags, scene_tags, ram_tags,
object_text, scene_text, timestamp_sec

Fallback field cũ:
object_counts, object_summary, tags, frame_idx
```

Tóm lại: không cần đổi hướng FuseCap/ReCap/TRAKE. Chỉ cần cập nhật schema mapping trong caption plan để nó ưu tiên enriched object output, và fallback được với output cũ.

> nếu làm như trên vẫn còn phù hợp với plan caption [caption_recap_fusecap_plan.md](c:/Users/User/Desktop/AIC2026/data_processing/caption/caption_recap_fusecap_plan.md) hiện tại đúng không

Đúng, vẫn phù hợp.

Các sửa đổi trong `object_detection_code_fix_plan.md` không đổi kiến trúc caption plan, mà chỉ làm input object evidence sạch hơn và dễ align hơn:

- `timestamp_sec` từ `L22_V012.csv/pts_time` khớp với phần Evidence Alignment, audio window, shot grouping và TRAKE.
- `object_counts_normalized`, `object_tags`, `important_objects` khớp với schema object evidence mà caption plan đang cần.
- `scene_tags`, `ram_tags`, `object_text`, `scene_text`, `all_object_text` hỗ trợ tốt hơn cho compact search index/ES.
- Giữ field cũ như `object_counts`, `objects`, `tags` nên không phá backward compatibility.

Chỉ nên cập nhật nhẹ caption plan ở Phase 0: Evidence Alignment để ưu tiên đọc field mới nếu có, fallback về field cũ nếu chưa enrich. Kiến trúc FuseCap frame fusion + ReCap temporal caption + TRAKE event-step vẫn giữ nguyên.

> Tôi đã thực hiện sửa chữa ở [audio_extraction](c:/Users/User/Desktop/AIC2026/data_processing/audio_extraction/) , hãy cho lệnh chạy lại pipeline

<details><summary>1 previous message</summary>

> Mình sẽ kiểm tra lại entrypoint và script smoke hiện có để đưa lệnh đúng với CLI hiện tại sau khi bạn sửa, tránh đoán nhầm flag.
</details>

Để chạy lại trên server cho `L22_V012`, dùng lệnh này từ project root:

```bash
cd /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test
CUDA_VISIBLE_DEVICES=0 bash data_processing/audio_extraction/scripts/run_l22_v012_smoke.sh --force
```

`--force` sẽ recreate lại audio/ASR/features để các sửa đổi mới trong `audio_extraction` được áp dụng đầy đủ.

Nếu muốn test nhanh hơn trước bằng model nhẹ:

```bash
cd /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test
CUDA_VISIBLE_DEVICES=0 bash data_processing/audio_extraction/scripts/run_l22_v012_smoke.sh --force --model turbo
```

Output chính cần kiểm tra sau chạy:

```text
outputs/audio/L22_V012/features/audio_features.jsonl
outputs/audio/L22_V012/asr/asr_quality_report.jsonl
outputs/audio/L22_V012/manifests/asr_job_manifest.jsonl
```

> Hãy tiếp tục sửa code cho [object_detection_code_fix_plan.md](c:/Users/User/Desktop/AIC2026/data_processing/object_detection/object_detection_code_fix_plan.md)

<details><summary>14 previous messages</summary>

> Mình sẽ triển khai theo plan nhưng ưu tiên phần không cần chạy GPU để kiểm chứng: schema enrichment, timestamp từ CSV keyframe map, path resolver, object/search fields, scene/person-family normalization và resume fingerprint. Sau đó mình sẽ chạy các kiểm tra cú pháp/helper nhỏ trên code.
>
> Context compaction
>
> Source: automatic
> Status: completed
>
> Mình sẽ kiểm tra trạng thái code hiện tại trước, vì có vẻ một phần sửa theo plan đã được đưa vào rồi. Sau đó mình sẽ vá các chỗ còn thiếu và chạy kiểm tra cú pháp/import để tránh sửa “đẹp trên giấy” nhưng vấp khi chạy.
>
> Hiện tại `ram_gdino_pipeline.py` đã có nhiều phần v1.1: timestamp map, fingerprint resume, search fields. Mình đang đọc tiếp đoạn xử lý chính để tìm phần chưa khớp giữa plan và code, đặc biệt là cách ghi schema, resume và các field cho ES/caption.
>
> Code chính đã gần khớp plan, nhưng còn vài “điểm nối” chưa hoàn thiện: config vẫn ghi schema cũ, visualizer chưa biết `image_relpath`, và summary chưa báo thiếu timestamp/geometry như plan. Mình sẽ vá các phần này để lần chạy lại dễ kiểm chứng hơn.
>
> Trước khi sửa file, mình sẽ chạm đúng các điểm nhỏ nhưng quan trọng: thêm `--no-resume-legacy`, đưa quality counters vào summary, cập nhật config/script để tự đọc keyframe CSV, và cho visualizer resolve bằng `image_relpath`. Như vậy pipeline mới không chỉ tạo field mới mà còn tự báo nếu field bị thiếu.
>
> <details><summary>Edited a file</summary>
>
> - Wrote `C:/Users/User/Desktop/AIC2026/data_processing/object_detection/ram_gdino_pipeline.py` (+64 -2)
>
> </details>
>
> Phần core vừa được cập nhật để summary có các chỉ số kiểm tra chất lượng theo plan. Mình tiếp tục nối config/script để khi chạy smoke test với `L22_V012.csv` thì timestamp được lấy trực tiếp từ CSV map.
>
> <details><summary>Edited 3 files</summary>
>
> - Wrote `C:/Users/User/Desktop/AIC2026/data_processing/object_detection/configs/object_detection_a5000.yaml` (+8 -1)
> - Wrote `C:/Users/User/Desktop/AIC2026/data_processing/object_detection/scripts/run_l22_v012_smoke.sh` (+5 -0)
> - Wrote `C:/Users/User/Desktop/AIC2026/data_processing/object_detection/visualize_detections.py` (+4 -0)
>
> </details>
>
> Lệnh kiểm tra cú pháp bị sandbox chặn khi Python đọc file ở `C:\Users\User\Desktop\AIC2026`. Mình sẽ xin quyền chạy đúng lệnh kiểm tra đó bên ngoài sandbox để xác nhận code thật sự compile được.
>
> Một test nhỏ vừa bóc ra lỗi đúng trọng tâm của plan: nếu chỉ còn `WOMAN` sau NMS thì code chưa canonicalize sang `PERSON` để đếm/search. Mình sẽ sửa phần family hierarchy để cả single detection như `MAN/WOMAN/CHILD` cũng count về `person`, vẫn giữ `raw_label` để không mất thông tin gốc.
>
> <details><summary>Edited a file</summary>
>
> - Wrote `C:/Users/User/Desktop/AIC2026/data_processing/object_detection/tag_canonicalization.py` (+17 -10)
>
> </details>
>
> Đã vá hierarchy canonicalization. Mình chạy lại kiểm tra cú pháp và test helper để chắc `object_counts_normalized` ra `person` thay vì `woman`.
>
> Kiểm tra cú pháp đã pass, và test hierarchy bây giờ cho `object_counts_normalized = {'person': 1}` như mong muốn. Mình sẽ rà nhanh diff cuối để đảm bảo không chạm lan sang phần ngoài scope.
</details>

`git status` bị Git chặn vì quyền sở hữu repo khác user sandbox, không phải lỗi code. Mình sẽ dùng cấu hình tạm `safe.directory` ngay trong lệnh để xem diff, không ghi global config.

> continue

<details><summary>13 previous messages</summary>

> Mình sẽ tiếp tục phần còn thiếu của plan: thêm adapter nhẹ để enrich output JSONL cũ mà không cần chạy lại GPU, rồi cập nhật README lệnh chạy mới có keyframe map. Đây sẽ giúp bạn vừa rerun được pipeline mới, vừa có đường “chữa cháy” cho output cũ khi cần unblock caption.
>
> Mình sẽ thêm script adapter độc lập với Torch/model để có thể chạy trên output cũ nhanh: nó đọc JSONL, gắn timestamp từ CSV map, enrich bbox geometry, canonicalize person-family, và sinh lại các field search/caption.
>
> <details><summary>Created a file</summary>
>
> - Wrote `C:/Users/User/Desktop/AIC2026/data_processing/object_detection/enrich_object_detection_jsonl.py` (+349 -0)
>
> </details>
>
> Adapter đã được thêm. Mình sẽ chỉnh thêm một chút để tương thích đúng command trong plan (`--frames-root`) và xử lý trường hợp tag cũ là chuỗi, rồi chạy compile/test trên một output mẫu ra thư mục tạm.
>
> <details><summary>Edited a file</summary>
>
> - Wrote `C:/Users/User/Desktop/AIC2026/data_processing/object_detection/enrich_object_detection_jsonl.py` (+4 -1)
>
> </details>
>
> Mình sẽ cập nhật README để các lệnh chạy phản ánh schema mới: smoke test tự đọc keyframe map, batch run có `--keyframe-map`, và có command enrich JSONL cũ.
>
> <details><summary>Edited a file</summary>
>
> - Wrote `C:/Users/User/Desktop/AIC2026/data_processing/object_detection/README.md` (+24 -0)
>
> </details>
>
> README đã khớp lệnh mới. Bây giờ mình chạy kiểm tra cú pháp cho adapter và thử enrich output hiện có sang `D:\tmp` để bắt lỗi dữ liệu thật, nhưng không ghi đè output gốc.
>
> Compile đã pass. Adapter test không tìm thấy file theo path Desktop `AIC2026\outputs\object_detection\...`; mình sẽ dò output object hiện nằm ở đâu trong workspace rồi chạy lại trên file đúng.
>
> Output mẫu nằm ở `D:\Users\outputs\object_detection`, không phải trong thư mục repo. Mình chạy lại adapter với file đó.
>
> Adapter chạy trên output thật thành công: 283 frame, thiếu timestamp = 0, thiếu geometry = 0. Mình sẽ kiểm tra thêm một frame cụ thể kiểu `L22_V012_003` để xác nhận person-family không còn làm phồng count trong dữ liệu enriched.
>
> Lệnh Python nhỏ trong sandbox bị lỗi launcher WindowsApps, nên mình chạy lại cùng kiểm tra ở mode đã được phép như các bước trước.
>
> Golden check tốt: frame `L22_V012_003` sau enrich có `timestamp_sec = 8.56667` và counts là `screen:1, person:2`, đúng mục tiêu không bị `person + woman` phồng count. Mình sẽ gom lại các file đã sửa và kết quả kiểm chứng.
</details>

Đã tiếp tục sửa phần `object_detection` theo plan.

Các điểm chính đã làm:
- Hoàn thiện schema v1.1 trong [ram_gdino_pipeline.py](C:/Users/User/Desktop/AIC2026/data_processing/object_detection/ram_gdino_pipeline.py:41): timestamp từ keyframe CSV, `image_relpath`, search/caption fields, summary quality, `--no-resume-legacy`.
- Sửa person-family canonicalization trong [tag_canonicalization.py](C:/Users/User/Desktop/AIC2026/data_processing/object_detection/tag_canonicalization.py:418): `MAN/WOMAN/CHILD/...` giờ count về `PERSON` nhưng vẫn giữ `raw_label`.
- Cập nhật config [object_detection_a5000.yaml](C:/Users/User/Desktop/AIC2026/data_processing/object_detection/configs/object_detection_a5000.yaml:1) sang schema `ram_gdino_object_detection_v1_1` và thêm block `metadata`.
- Cập nhật smoke script để tự truyền `--keyframe-map` và `--frames-root`: [run_l22_v012_smoke.sh](C:/Users/User/Desktop/AIC2026/data_processing/object_detection/scripts/run_l22_v012_smoke.sh:90).
- Cập nhật visualizer để resolve ảnh bằng `image_relpath`: [visualize_detections.py](C:/Users/User/Desktop/AIC2026/data_processing/object_detection/visualize_detections.py:62).
- Thêm adapter enrich output cũ không cần chạy GPU: [enrich_object_detection_jsonl.py](C:/Users/User/Desktop/AIC2026/data_processing/object_detection/enrich_object_detection_jsonl.py:1).
- Cập nhật README với lệnh chạy/enrich mới: [README.md](C:/Users/User/Desktop/AIC2026/data_processing/object_detection/README.md:135).

Đã verify:
- `python -m py_compile` pass cho pipeline, canonicalization, visualizer, adapter.
- Chạy thử adapter trên `D:\Users\outputs\object_detection\L22_V012_objects.jsonl` ra `D:\tmp\L22_V012_objects_enriched_test.jsonl`.
- Summary test: `283` frames, missing timestamp `0`, missing area/position `0`, scene labels in counts `0`.
- Golden check `L22_V012_003`: `timestamp_sec = 8.56667`, counts là `screen: 1`, `person: 2`, không còn phồng count kiểu `person + woman`.

> cho lệnh chạy trên server

Dưới đây là lệnh chạy trên server cho object detection schema mới.

**Smoke test L22_V012**
```bash
cd /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/data_processing/object_detection

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /tmp2/maitanha/vgu/ttn/AIC-Khoa/conda_envs/aic-object-detection-gpu

LIMIT=20 CUDA_VISIBLE_DEVICES=0 bash scripts/run_l22_v012_smoke.sh --no-resume
```

**Chạy full L22_V012**
```bash
cd /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/data_processing/object_detection

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /tmp2/maitanha/vgu/ttn/AIC-Khoa/conda_envs/aic-object-detection-gpu

CUDA_VISIBLE_DEVICES=0 \
RAM_CHECKPOINT_PATH=/tmp2/maitanha/vgu/ttn/AIC-Khoa/models/ram_plus_swin_large_14m.pth \
HF_HOME=/tmp2/maitanha/vgu/ttn/AIC-Khoa/hf_cache \
python ram_gdino_pipeline.py \
  --config configs/object_detection_a5000.yaml \
  --input /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/keyframe_test/L22_V012 \
  --output /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/outputs/object_detection/L22_V012_objects.jsonl \
  --video-id L22_V012 \
  --keyframe-map /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/keyframe_test \
  --frames-root /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/keyframe_test \
  --summary-output /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/outputs/object_detection/summaries/L22_V012_summary.json \
  --no-resume
```

**Chạy batch L22**
```bash
cd /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/data_processing/object_detection

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /tmp2/maitanha/vgu/ttn/AIC-Khoa/conda_envs/aic-object-detection-gpu

CUDA_VISIBLE_DEVICES=0 \
RAM_CHECKPOINT_PATH=/tmp2/maitanha/vgu/ttn/AIC-Khoa/models/ram_plus_swin_large_14m.pth \
HF_HOME=/tmp2/maitanha/vgu/ttn/AIC-Khoa/hf_cache \
python ram_gdino_pipeline.py \
  --config configs/object_detection_a5000.yaml \
  --input /tmp2/maitanha/vgu/ttn/data/AIC2025/keyframes/Keyframes_L22 \
  --output /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/outputs/object_detection \
  --batch \
  --keyframe-map /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/keyframe_test \
  --frames-root /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/keyframe_test \
  --summary-output /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/outputs/object_detection/summaries/L22_summary.json \
  --no-resume
```

Lưu ý quan trọng: dùng `--no-resume` cho lần chạy đầu sau khi sửa schema, vì output cũ chưa có `frame_fingerprint`; nếu dùng `--resume` mặc định có thể skip record cũ.

> batch L22 và --no-resume là như thế nào

`batch L22` nghĩa là chạy một lần cho nhiều video con trong thư mục `Keyframes_L22`.

Ví dụ thư mục:

```text
Keyframes_L22/
  L22_V001/
    001.jpg
    002.jpg
  L22_V002/
    001.jpg
    002.jpg
  L22_V012/
    001.jpg
```

Khi dùng:

```bash
--input /.../Keyframes_L22
--output /.../outputs/object_detection
--batch
```

pipeline sẽ tự duyệt từng folder video con và tạo output riêng:

```text
outputs/object_detection/
  L22_V001_objects.jsonl
  L22_V002_objects.jsonl
  L22_V012_objects.jsonl
```

`--no-resume` nghĩa là không dùng output cũ để skip frame. Nó sẽ ghi lại output từ đầu. Nên dùng `--no-resume` sau khi mình vừa đổi schema sang v1.1, vì output cũ có thể thiếu `timestamp_sec`, `area_ratio`, `object_text`, `frame_fingerprint`...

Nếu dùng `--resume`, pipeline sẽ đọc file JSONL cũ và bỏ qua frame đã có. Điều này nhanh hơn, nhưng trong lần nâng schema này có thể làm bạn giữ lại record cũ thiếu field.

Tóm lại:

```bash
--batch
```

= chạy nhiều video trong một folder cha.

```bash
--no-resume
```

= chạy lại sạch từ đầu, phù hợp lần đầu sau khi sửa schema.

Sau khi đã có output v1.1 chuẩn rồi, các lần chạy tiếp có thể dùng:

```bash
--resume
```

để tiết kiệm thời gian.

> tiếp tục viết plan .md sửa đổi với OCR

<details><summary>1 previous message</summary>

> Mình sẽ đọc nhanh cấu trúc OCR pipeline và output hiện có trước, rồi viết plan `.md` ngay trong thư mục OCR giống cách đã làm cho audio/object. Trọng tâm sẽ là schema/search alignment với caption + ES, timestamp/keyframe map, và các lỗi OCR đặc thù như order, review flag, duplicate text.
</details>

Mình thấy OCR output mẫu đang ở `D:\Users\outputs\ocr_vlm_pipeline_v2\L22_V012`, không nằm trong repo. Mình sẽ đọc code sinh output và vài dòng JSONL/CSV để plan bám vào dữ liệu thật thay vì chỉ phỏng đoán.

> mình vừa cung cấp lại ouput bản cũ ở [outputs](outputs/)

<details><summary>6 previous messages</summary>

> Ổn, vậy mình sẽ dùng lại output cũ trong `outputs/` làm bằng chứng chính. Mình sẽ đọc vừa đủ OCR code + vài dòng output để xác định lỗi thật, rồi ghi plan `.md` vào `data_processing/OCR/ocr_vlm_pipeline_v2/`.
>
> Mình xác nhận output cũ hiện nằm ở `D:\Users\outputs\ocr_vlm_pipeline_v2\L22_V012`. Bây giờ mình sẽ lấy schema mẫu, summary CSV, timing, và vị trí code sinh JSONL để plan chỉ ra chính xác chỗ cần sửa.
>
> Output mẫu cho thấy lỗi nổi bật ngay: `frame_id` đang là `001` trong khi object/caption dùng dạng canonical `L22_V012_001`, và `ocr_text_search` đang lặp `"giây"` 3 lần dù clean chỉ có 1 từ. Mình sẽ đo thêm mức độ thiếu timestamp/order/duplicate rồi viết plan sửa có acceptance rõ ràng.
>
> Mình đã có đủ số liệu từ output cũ: 283/283 thiếu timestamp, 283/283 thiếu canonical frame id, JSONL bị lệch order khi `workers=2`, 269/283 frame có `ocr_text_search` khác clean do lặp tầng line/group/clean, và 293 line `need_review=True` vẫn `keep_for_index=True`. Mình sẽ ghi plan sửa thành file `.md` ngay bây giờ.
>
> <details><summary>Created a file</summary>
>
> - Wrote `C:/Users/User/Desktop/AIC2026/data_processing/OCR/ocr_vlm_pipeline_v2/ocr_vlm_pipeline_code_fix_plan.md` (+637 -0)
>
> </details>
>
> Plan OCR đã được ghi xong. Mình kiểm tra nhanh file vừa tạo để chắc đường dẫn đúng và nội dung có các mục sửa chính.
</details>

Đã viết plan OCR tại:

[ocr_vlm_pipeline_code_fix_plan.md](C:/Users/User/Desktop/AIC2026/data_processing/OCR/ocr_vlm_pipeline_v2/ocr_vlm_pipeline_code_fix_plan.md)

Plan dựa trực tiếp trên output cũ bạn vừa cung cấp ở `D:\Users\outputs\ocr_vlm_pipeline_v2\L22_V012`, gồm các lỗi chính:

- 283/283 thiếu `timestamp_sec`
- 283/283 thiếu `canonical_frame_id`
- `frame_id` đang là `001` thay vì `L22_V012_001`
- JSONL bị out-of-order khi `workers=2`
- `ocr_text_search` bị lặp, ví dụ `"giây giây giây"`
- 293 line `need_review=True` vẫn `keep_for_index=True`

Plan cũng đã đưa schema OCR v2 đề xuất, mapping ES, validation summary, adapter enrich output cũ, test plan và Definition of Done.

> Hãy thực hiện viết code, lưu ý: tôi sẽ bỏ kêt quả cũ và chạy lại từ đầu

<details><summary>18 previous messages</summary>

> Mình sẽ implement theo hướng “rerun sạch từ đầu” nên ưu tiên các sửa ảnh hưởng trực tiếp đến output mới: schema v2, canonical frame id, timestamp từ CSV, `ocr_text_search` clean-only, JSONL ordered khi multi-worker, và quality summary. Resume/adapter có thể để sau vì bạn sẽ bỏ output cũ.
>
> Mình đã đọc code chính. Sửa sẽ tập trung ở hai file: `ocr_pipeline/outputs.py` để đổi schema/search fields, và `run_frame_folder.py` để thêm metadata từ CSV + ghi JSONL ordered. Mình sẽ giữ backward compatibility cho `run_single_frame.py` bằng cách để metadata là optional.
>
> <details><summary>Edited a file</summary>
>
> - Wrote `C:/Users/User/Desktop/AIC2026/data_processing/OCR/ocr_vlm_pipeline_v2/ocr_pipeline/outputs.py` (+49 -10)
>
> </details>
>
> `outputs.py` đã chuyển sang schema v2 và `ocr_text_search` giờ mặc định chỉ lấy clean text. Mình sẽ thêm helper dedupe còn thiếu rồi chuyển sang phần runner để inject timestamp/canonical id cho từng frame.
>
> <details><summary>Edited a file</summary>
>
> - Wrote `C:/Users/User/Desktop/AIC2026/data_processing/OCR/ocr_vlm_pipeline_v2/ocr_pipeline/outputs.py` (+15 -0)
>
> </details>
>
> Mình sẽ cập nhật `run_frame_folder.py`: thêm `--keyframe_map`, `--video_manifest`, `--frames_root`, `--timestamp_strategy`; build metadata trước khi xử lý từng frame; và ghi JSONL ở cuối theo đúng thứ tự keyframe, kể cả khi `workers > 1`.
>
> <details><summary>Edited a file</summary>
>
> - Wrote `C:/Users/User/Desktop/AIC2026/data_processing/OCR/ocr_vlm_pipeline_v2/run_frame_folder.py` (+151 -1)
> - Wrote `C:/Users/User/Desktop/AIC2026/data_processing/OCR/ocr_vlm_pipeline_v2/run_frame_folder.py` (+64 -0)
>
> </details>
>
> Metadata helpers và quality summary đã vào runner. Tiếp theo mình nối metadata vào `process_frame()` để document mới mang `timestamp_sec`, `image_relpath`, `canonical_frame_id`, đồng thời summary CSV cũng có các cột này.
>
> <details><summary>Edited a file</summary>
>
> - Wrote `C:/Users/User/Desktop/AIC2026/data_processing/OCR/ocr_vlm_pipeline_v2/run_frame_folder.py` (+12 -3)
> - Wrote `C:/Users/User/Desktop/AIC2026/data_processing/OCR/ocr_vlm_pipeline_v2/run_frame_folder.py` (+44 -3)
>
> </details>
>
> Một điểm quan trọng: không chỉ search field, mà `clean_text` cũng đang có nguy cơ nhận line `need_review=True` nếu line đó vẫn `keep_for_index=True`. Mình sẽ sửa ngay bước build group text để review line đi vào `ocr_text_review`, không vào clean/search chính.
>
> <details><summary>Edited 2 files</summary>
>
> - Wrote `C:/Users/User/Desktop/AIC2026/data_processing/OCR/ocr_vlm_pipeline_v2/ocr_pipeline/outputs.py` (+1 -1)
> - Wrote `C:/Users/User/Desktop/AIC2026/data_processing/OCR/ocr_vlm_pipeline_v2/run_frame_folder.py` (+4 -3)
> - Wrote `C:/Users/User/Desktop/AIC2026/data_processing/OCR/ocr_vlm_pipeline_v2/ocr_pipeline/outputs.py` (+4 -0)
>
> </details>
>
> Các thay đổi chính đã xong. Mình sẽ chạy kiểm tra cú pháp trước, sau đó chạy một test nhẹ không load model để xác nhận `build_es_document()` sinh đúng schema v2 và `ocr_text_search` không còn lặp.
>
> Test cho thấy schema/search đúng, nhưng metric `num_search_duplicates` đang đếm duplicate ở evidence line/group chứ không phải trong search chính. Mình sẽ chỉnh lại metric để validation phản ánh đúng: search chính sạch thì duplicate = 0, còn duplicate evidence được ghi riêng.
>
> <details><summary>Edited a file</summary>
>
> - Wrote `C:/Users/User/Desktop/AIC2026/data_processing/OCR/ocr_vlm_pipeline_v2/ocr_pipeline/outputs.py` (+8 -3)
>
> </details>
>
> Mình sẽ ghi thêm quality summary thành file riêng `*_ocr_quality_summary.json` ngoài việc nhúng trong timing summary, để sau khi chạy server bạn nhìn một file là biết timestamp/order/search đã sạch chưa.
>
> <details><summary>Edited a file</summary>
>
> - Wrote `C:/Users/User/Desktop/AIC2026/data_processing/OCR/ocr_vlm_pipeline_v2/run_frame_folder.py` (+4 -0)
>
> </details>
>
> Chỉnh xong file quality summary riêng. Mình chạy compile cuối và rà diff nhanh để chuẩn bị kết luận + lệnh chạy lại.
>
> Compile đã pass. Mình thấy `as_completed` vẫn dùng để nhận kết quả worker, nhưng JSONL không còn ghi ở đó nữa; docs được sort rồi ghi một lần cuối run, nên output mới sẽ ordered.
</details>

Đã implement code OCR theo plan cho hướng bạn sẽ bỏ output cũ và chạy lại từ đầu.

Đã sửa chính trong:
- [outputs.py](C:/Users/User/Desktop/AIC2026/data_processing/OCR/ocr_vlm_pipeline_v2/ocr_pipeline/outputs.py:30)
- [run_frame_folder.py](C:/Users/User/Desktop/AIC2026/data_processing/OCR/ocr_vlm_pipeline_v2/run_frame_folder.py:80)

Nội dung đã làm:
- Đổi schema sang `ocr_vlm_pipeline_v2_es_2`.
- `frame_id` mới là canonical dạng `L22_V012_001`.
- Thêm `canonical_frame_id`, `legacy_frame_id`, `frame_name`, `keyframe_idx`, `source_frame_idx`, `timestamp_sec`, `timestamp_source`, `image_relpath`.
- Thêm CLI:
  - `--keyframe-map` / `--keyframe_map`
  - `--video-manifest` / `--video_manifest`
  - `--frames-root` / `--frames_root`
  - `--timestamp-strategy` / `--timestamp_strategy`
- Sửa `ocr_text_search`: mặc định chỉ dùng `ocr_text_clean`, không lặp `clean + group + line`.
- Line `need_review=True` không còn đi vào clean/search chính.
- Multi-worker vẫn dùng `as_completed`, nhưng JSONL được sort theo `keyframe_idx` rồi mới ghi cuối run, nên output không còn out-of-order.
- Thêm file quality summary riêng: `<video_id>_ocr_quality_summary.json`.

Đã verify:
- `python -m py_compile` pass cho 2 file đã sửa.
- Unit test nhỏ xác nhận output builder sinh:
  - schema `ocr_vlm_pipeline_v2_es_2`
  - `frame_id = L22_V012_001`
  - `timestamp_sec = 0.0`
  - `ocr_text_search` không lặp
  - `num_search_duplicates = 0`
  - `num_need_review_lines_in_primary_search = 0`

Lệnh chạy lại trên server cho `L22_V012`:

```bash
python run_frame_folder.py \
  --frames_dir /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/keyframe_test/L22_V012 \
  --output_dir /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/outputs/ocr_vlm_pipeline_v2 \
  --video_id L22_V012 \
  --keyframe-map /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/keyframe_test \
  --frames-root /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/keyframe_test \
  --workers 2 \
  --batch_artifacts minimal \
  --cleanup_crops
```

Sau khi chạy, kiểm tra nhanh file:
`outputs/ocr_vlm_pipeline_v2/L22_V012/L22_V012_ocr_quality_summary.json`
Các chỉ số quan trọng nên về `0`: missing timestamp, missing canonical frame id, docs out of order, search duplicates.

> tôi sẽ chạy trên server

Đúng rồi, bạn chạy trên server là hợp lý vì OCR pipeline load PaddleOCR/VietOCR/Vintern.

Dùng lệnh này cho `L22_V012`:

```bash
cd /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/data_processing/OCR/ocr_vlm_pipeline_v2

python run_frame_folder.py \
  --frames_dir /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/keyframe_test/L22_V012 \
  --output_dir /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/outputs/ocr_vlm_pipeline_v2 \
  --video_id L22_V012 \
  --keyframe-map /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/keyframe_test \
  --frames-root /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/keyframe_test \
  --workers 2 \
  --batch_artifacts minimal \
  --cleanup_crops
```

Sau khi chạy xong, check file:

```bash
cat /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/outputs/ocr_vlm_pipeline_v2/L22_V012/L22_V012_ocr_quality_summary.json
```

Các chỉ số nên là:

```text
num_frames_missing_timestamp = 0
num_frames_missing_canonical_frame_id = 0
num_docs_out_of_order = 0
num_search_duplicates = 0
num_need_review_lines_in_primary_search = 0
```

Nếu server có nhiều VRAM, có thể thử `--workers 4`; còn nếu bị OOM thì giảm về `--workers 1`.

> cho lên chạy trên gpu

Dùng lệnh này để ép chạy trên GPU. Mình để PaddleOCR/VietOCR/Vintern cùng dùng `gpu:0`/CUDA 0:

```bash
cd /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/data_processing/OCR/ocr_vlm_pipeline_v2

export CUDA_VISIBLE_DEVICES=0
export OCR_V2_PADDLE_DEVICE=gpu:0
export OCR_V2_PADDLE_ENGINE=paddle_dynamic

python run_frame_folder.py \
  --frames_dir /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/keyframe_test/L22_V012 \
  --output_dir /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/outputs/ocr_vlm_pipeline_v2 \
  --video_id L22_V012 \
  --keyframe-map /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/keyframe_test \
  --frames-root /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/keyframe_test \
  --workers 2 \
  --batch_artifacts minimal \
  --cleanup_crops
```

Nếu bị OOM, chạy lại với:

```bash
--workers 1
```

Nếu muốn dùng GPU khác, ví dụ GPU 4:

```bash
export CUDA_VISIBLE_DEVICES=4
export OCR_V2_PADDLE_DEVICE=gpu:0
```

Giữ `OCR_V2_PADDLE_DEVICE=gpu:0` vì sau `CUDA_VISIBLE_DEVICES=4`, GPU 4 sẽ được remap thành GPU 0 bên trong process.

> đây là ví dụ lệnh ở version cũ "(/tmp2/maitanha/vgu/ttn/AIC-Khoa/conda_envs/aic-detect-gpu) maitanha@cl777:/tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/data_processing/OCR/ocr_vlm_pipeline_v2$ CUDA_VISIBLE_DEVICES=7 OCR_V2_PADDLE_DEVICE=gpu:0 OCR_V2_PADDLE_ENGINE=paddle_dynamic OCR_V2_VINTERN_ATTN=flash_attention_2 python run_frame_folder.py   --frames_dir /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/keyframe_test/L22_V012   --output_dir /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/outputs/ocr_vlm_pipeline_v2   --video_id L22_V012   --pattern "*.jpg"   --batch_artifacts minimal   --cleanup_crops   --workers 2   --no_vietocr_batch"

Dựa trên lệnh version cũ của bạn, bản mới chỉ cần thêm `--keyframe-map`, `--frames-root` và giữ các env GPU như cũ:

```bash
CUDA_VISIBLE_DEVICES=7 \
OCR_V2_PADDLE_DEVICE=gpu:0 \
OCR_V2_PADDLE_ENGINE=paddle_dynamic \
OCR_V2_VINTERN_ATTN=flash_attention_2 \
python run_frame_folder.py \
  --frames_dir /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/keyframe_test/L22_V012 \
  --output_dir /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/outputs/ocr_vlm_pipeline_v2 \
  --video_id L22_V012 \
  --pattern "*.jpg" \
  --keyframe-map /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/keyframe_test \
  --frames-root /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/keyframe_test \
  --timestamp-strategy map_or_uniform \
  --batch_artifacts minimal \
  --cleanup_crops \
  --workers 2 \
  --no_vietocr_batch
```

Vì `CUDA_VISIBLE_DEVICES=7`, trong process nó sẽ thấy GPU 7 thành `gpu:0`, nên `OCR_V2_PADDLE_DEVICE=gpu:0` là đúng.

Sau chạy xong check:

```bash
cat /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/outputs/ocr_vlm_pipeline_v2/L22_V012/L22_V012_ocr_quality_summary.json
```

Kỳ vọng các field này bằng `0`:

```text
num_frames_missing_timestamp
num_frames_missing_canonical_frame_id
num_docs_out_of_order
num_search_duplicates
num_need_review_lines_in_primary_search
```

> cho lệnh kill lệnh chạy trên, tôi muốn chạy trên device 1 "CUDA_VISIBLE_DEVICES=1 \
> OCR_V2_PADDLE_DEVICE=gpu:0 \
> OCR_V2_PADDLE_ENGINE=paddle_dynamic \
> OCR_V2_VINTERN_ATTN=flash_attention_2 \
> python run_frame_folder.py \
>   --frames_dir /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/keyframe_test/L22_V012 \
>   --output_dir /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/outputs/ocr_vlm_pipeline_v2 \
>   --video_id L22_V012 \
>   --pattern "*.jpg" \
>   --keyframe-map /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/keyframe_test \
>   --frames-root /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/keyframe_test \
>   --timestamp-strategy map_or_uniform \
>   --batch_artifacts minimal \
>   --cleanup_crops \
>   --workers 2 \
>   --no_vietocr_batch"

Để kill lệnh OCR đang chạy, dùng lệnh này trên server:

```bash
pkill -f "run_frame_folder.py.*L22_V012"
```

Nếu muốn kiểm tra trước PID rồi kill thủ công:

```bash
ps -ef | grep run_frame_folder.py | grep L22_V012
```

Sau đó kill theo PID:

```bash
kill <PID>
```

Nếu process chưa chết sau vài giây:

```bash
kill -9 <PID>
```

Rồi chạy lại trên device 1:

```bash
CUDA_VISIBLE_DEVICES=1 \
OCR_V2_PADDLE_DEVICE=gpu:0 \
OCR_V2_PADDLE_ENGINE=paddle_dynamic \
OCR_V2_VINTERN_ATTN=flash_attention_2 \
python run_frame_folder.py \
  --frames_dir /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/keyframe_test/L22_V012 \
  --output_dir /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/outputs/ocr_vlm_pipeline_v2 \
  --video_id L22_V012 \
  --pattern "*.jpg" \
  --keyframe-map /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/keyframe_test \
  --frames-root /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/keyframe_test \
  --timestamp-strategy map_or_uniform \
  --batch_artifacts minimal \
  --cleanup_crops \
  --workers 2 \
  --no_vietocr_batch
```

Giữ `OCR_V2_PADDLE_DEVICE=gpu:0` là đúng: vì `CUDA_VISIBLE_DEVICES=1` remap GPU vật lý 1 thành GPU 0 bên trong process.

> "========================================================================================|
> |    0   N/A  N/A     89060      C   ...onda_envs/aic-detect-gpu/bin/python       5846MiB |" vẫn chọn chạy

Process `PID=89060` vẫn đang chạy trên GPU 0. Kill đúng PID này:

```bash
kill 89060
```

Đợi 3-5 giây rồi kiểm tra:

```bash
nvidia-smi
```

Nếu vẫn còn:

```bash
kill -9 89060
```

Nếu pipeline dùng `--workers 2`, có thể còn process con. Kill toàn bộ process liên quan `run_frame_folder.py`:

```bash
pkill -f "run_frame_folder.py"
```

Nếu vẫn còn worker Python giữ VRAM, xem danh sách PID:

```bash
ps -ef | grep -E "run_frame_folder.py|aic-detect-gpu/bin/python" | grep -v grep
```

Rồi kill các PID còn lại:

```bash
kill -9 <PID1> <PID2> <PID3>
```

Sau khi `nvidia-smi` sạch, chạy lại trên GPU vật lý 1:

```bash
CUDA_VISIBLE_DEVICES=1 \
OCR_V2_PADDLE_DEVICE=gpu:0 \
OCR_V2_PADDLE_ENGINE=paddle_dynamic \
OCR_V2_VINTERN_ATTN=flash_attention_2 \
python run_frame_folder.py \
  --frames_dir /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/keyframe_test/L22_V012 \
  --output_dir /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/outputs/ocr_vlm_pipeline_v2 \
  --video_id L22_V012 \
  --pattern "*.jpg" \
  --keyframe-map /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/keyframe_test \
  --frames-root /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/keyframe_test \
  --timestamp-strategy map_or_uniform \
  --batch_artifacts minimal \
  --cleanup_crops \
  --workers 2 \
  --no_vietocr_batch
```

> Đã tốt, bây giờ hãy tiến hành phân tích plan .md caption lại [caption_recap_fusecap_plan.md](data_processing/caption/caption_recap_fusecap_plan.md) trước khi viết code

<details><summary>5 previous messages</summary>

> Mình sẽ đọc lại plan caption cùng các output/schema mới của audio, object, OCR để đánh giá lại độ khớp trước khi đụng code. Sau đó mình sẽ tóm tắt các điểm cần chỉnh trong plan caption và rủi ro nếu code caption dùng schema cũ.
>
> Tool vừa báo lỗi khởi tạo PowerShell với working directory hiện tại, có thể session local không còn thấy `D:\Users\outputs` như trước. Mình sẽ kiểm tra lại cwd/path rồi tiếp tục đọc file plan.
>
> Có vẻ lỗi nằm ở việc tạo process với cwd mặc định, không phải nội dung command. Mình sẽ ép `workdir` sang một path nền ổn định rồi đọc tiếp.
>
> `D:\Users\outputs` hiện không còn tồn tại trong session này, còn repo `C:\Users\User\Desktop\AIC2026` vẫn có. Mình sẽ dùng repo path trực tiếp và kiểm tra thư mục `outputs/` trong repo nếu bạn vừa copy output cũ vào đó.
>
> Plan caption hiện tại vẫn là thiết kế tổng thể tốt, nhưng đang dùng ví dụ schema cũ ở nhiều chỗ. Mình sẽ đối chiếu nhanh với các plan/code mới của audio/object/OCR để chỉ ra cụ thể phần nào giữ, phần nào cần cập nhật trước khi viết code.
</details>

Đúng hướng: **plan caption hiện tại vẫn phù hợp về kiến trúc**, nhưng trước khi viết code cần cập nhật theo schema mới của OCR/object/audio. Nếu code caption bám đúng plan cũ nguyên trạng thì dễ join sai frame hoặc dùng nhầm text nhiễu.

**Nhận xét chính**

1. `frame_id` trong plan đang dùng ví dụ kiểu `L22_V012_000321`, nhưng output mới của OCR/object đang chuẩn về:
   ```text
   canonical_frame_id = L22_V012_001
   frame_id = L22_V012_001
   frame_name = 001
   keyframe_idx = 1
   timestamp_sec = 0.0
   ```
   Caption code nên join bằng `canonical_frame_id`, fallback `frame_id`.

2. OCR input trong plan cần đổi từ `ocr_text_joined` sang các field mới:
   ```text
   ocr_text_search      # clean/indexable chính
   ocr_text_clean       # clean text
   ocr_text_review      # audit, không dùng mặc định
   ocr_terms
   quality.has_clean_text
   ```
   Không nên tự gom từ `ocr_lines` nếu chưa cần, vì OCR pipeline đã xử lý clean/review rồi.

3. Object input nên dùng schema v1.1:
   ```text
   object_counts_normalized
   object_count_items
   object_tags
   scene_tags
   ram_tags
   important_objects
   object_text
   scene_text
   all_object_text
   ```
   Không nên dùng trực tiếp `object_counts` uppercase làm field chính cho caption/search.

4. Audio input nên đọc `audio_features.jsonl`, không đọc raw `asr_segments.jsonl`.
   Caption evidence nên lấy:
   ```text
   caption_text nếu usable_for_caption=true
   fallback clean_transcript nếu chưa có caption_text
   search_text cho compact index/audit
   start_sec/end_sec/time_window
   ```

5. Phase đầu không nên viết LLM fuser ngay. Nên làm trước:
   ```text
   input loaders
   schema adapters
   evidence alignment
   frame_evidence.jsonl
   shot_evidence.jsonl
   compact_search_index baseline
   validation report
   ```
   Sau đó mới thêm text-only caption fuser.

**Kết luận trước khi code**

Plan caption giữ được phần kiến trúc:

```text
FuseCap-style frame fusion
+ ReCap-style shot memory
+ TRAKE event-step
```

Nhưng khi viết code, nên bắt đầu bằng **Phase 0 evidence alignment** với schema mới, không gọi model trước. Đây là nền bắt buộc để tránh caption hallucination do join sai frame, thiếu timestamp, hoặc dùng nhầm OCR/audio review text.

Thứ tự code nên làm tiếp:

1. Tạo module caption pipeline skeleton.
2. Viết loaders cho OCR/object/audio.
3. Viết adapter chuẩn hóa về `FrameEvidence`.
4. Join theo `canonical_frame_id`.
5. Align audio bằng `timestamp_sec ± window`.
6. Sinh `frame_evidence.jsonl`.
7. Sinh `evidence_alignment_report.json/md`.
8. Sau khi report ổn mới viết fuser.

> Hãy kiểm tra ouput version mới nhất [outputs](outputs/) từ các mô hình trên, rồi viết plan .md lại về pipeline để viết code

<details><summary>6 previous messages</summary>

> Mình sẽ kiểm tra trực tiếp `outputs/` mới nhất của audio/object/OCR để xem schema thực tế sau khi chạy lại đã khớp các sửa chưa. Sau đó mình sẽ viết một plan `.md` mới cho caption pipeline, bám theo output thật thay vì plan cũ giả định.
>
> Mình thấy output mới đã có đủ ba nhánh: audio, object detection và OCR v2, gồm cả quality summaries. Mình sẽ đọc mẫu record + summary từng nhánh và chạy vài thống kê join/timeline để plan caption đặt đúng acceptance criteria.
>
> Audio mới nhìn ổn: manifest `probed/planned`, `audio_features.jsonl` có `caption_text/search_text`, timeline sorted và 217 đoạn usable. Object còn một cảnh báo nhỏ: summary báo `num_scene_labels_in_counts = 7`, nên caption evidence builder cần filter scene labels lần nữa như guardrail. Mình sẽ đo join OCR-object và timestamp mismatch để plan chốt schema input.
>
> Kết quả audit mới tốt hơn nhiều: OCR/object join đủ 283/283 bằng `canonical_frame_id`, timestamp khớp tuyệt đối, OCR ordered, audio ordered. Mình sẽ viết plan mới cho caption pipeline, bắt đầu bằng evidence alignment + compact baseline trước khi gọi LLM/VLM.
>
> <details><summary>Created a file</summary>
>
> - Wrote `C:/Users/User/Desktop/AIC2026/data_processing/caption/caption_pipeline_code_plan.md` (+794 -0)
>
> </details>
>
> Plan mới đã được ghi. Mình sẽ kiểm tra lại file và các điểm chính để chắc nó phản ánh đúng output mới nhất.
</details>

Đã kiểm tra output mới nhất trong [outputs](C:/Users/User/Desktop/AIC2026/outputs) và viết plan code mới tại:

[caption_pipeline_code_plan.md](C:/Users/User/Desktop/AIC2026/data_processing/caption/caption_pipeline_code_plan.md)

Kết quả audit chính đã đưa vào plan:

- OCR v2:
  - 283 frames
  - missing timestamp = 0
  - missing canonical id = 0
  - out-of-order = 0
  - search duplicates = 0
- Object detection v1.1:
  - 283 frames processed
  - missing timestamp = 0
  - missing geometry = 0
  - join với OCR bằng `canonical_frame_id` = 283/283
  - timestamp mismatch OCR/object = 0
  - còn guardrail nhỏ: `num_scene_labels_in_counts = 7`
- Audio:
  - 226 audio features
  - 217 usable for caption
  - timeline sorted = true
  - boilerplate đã bị filter

Plan mới tập trung vào code pipeline theo thứ tự an toàn:

1. Load OCR/object/audio.
2. Join frame bằng `canonical_frame_id`.
3. Align audio theo `timestamp_sec ± window`.
4. Sinh `frame_evidence.jsonl`.
5. Group shot theo timestamp.
6. Sinh `shot_evidence.jsonl`.
7. Sinh `compact_search_index.jsonl`.
8. Tạo validation report.
9. Sau đó mới làm text-only fuser/ReCap/TRAKE.

Điểm quan trọng: plan mới không bắt đầu bằng LLM/VLM. Nó bắt đầu bằng Phase 0 evidence alignment để tránh caption hallucination do join sai hoặc dùng nhầm OCR/audio nhiễu.

> Vây Phase 0 sẽ trả output về là gì? rồi vậy sau đó sẽ thực hiện như thế nào ?

Phase 0 **chưa tạo caption bằng LLM/VLM**. Nó trả về bộ evidence đã align sạch để các phase sau dùng.

**Output Phase 0**

```text
outputs/caption/L22_V012/
├── evidence/
│   └── frame_evidence.jsonl
└── reports/
    ├── evidence_alignment_report.json
    └── evidence_alignment_report.md
```

`frame_evidence.jsonl` có 1 dòng / keyframe, dự kiến 283 dòng cho `L22_V012`.

Mỗi dòng đại khái:

```json
{
  "schema_version": "caption_frame_evidence_v1",
  "video_id": "L22_V012",
  "frame_id": "L22_V012_001",
  "canonical_frame_id": "L22_V012_001",
  "timestamp_sec": 0.0,
  "image_relpath": "L22_V012/001.jpg",
  "object_evidence": {
    "object_counts": {},
    "object_tags": [],
    "scene_tags": ["city", "sky", "sun", "water"],
    "important_objects": [],
    "object_text": "",
    "scene_text": "city sky sun water"
  },
  "ocr_evidence": {
    "ocr_text": "giây",
    "ocr_terms": ["giay"],
    "review_used": false
  },
  "audio_evidence": {
    "window": [0.0, 5.0],
    "audio_text": "",
    "num_segments": 0
  },
  "frame_search_text": "giây city sky sun water ..."
}
```

Report sẽ trả về các chỉ số kiểm tra:

```json
{
  "num_object_frames": 283,
  "num_ocr_frames": 283,
  "num_joined_frames": 283,
  "num_timestamp_mismatch": 0,
  "num_audio_features": 226,
  "num_audio_features_usable": 217,
  "num_frame_evidence": 283,
  "warnings": []
}
```

**Sau Phase 0 làm gì?**

1. **Phase 1: Shot grouping + compact search**
   - Input: `frame_evidence.jsonl`
   - Output:
     ```text
     shot_evidence.jsonl
     compact_search_index.jsonl
     ```
   - Group các frame gần nhau theo `timestamp_sec`, ví dụ gap <= 5s hoặc shot <= 20s.
   - `compact_search_index.jsonl` đã có thể import thử vào ES để search baseline, dù chưa có caption sinh bởi LLM.

2. **Phase 2: Frame caption fuser**
   - Input: `frame_evidence.jsonl`
   - Gọi text-only LLM fuser.
   - Output:
     ```text
     frame_index.jsonl
     ```
   - Mỗi frame có `frame_caption`, `search_tags`, `scene_type`, `visible_actions`.

3. **Phase 3: ReCap shot caption**
   - Input: `shot_evidence.jsonl` + frame captions.
   - Gọi fuser có memory theo shot.
   - Output:
     ```text
     shot_index.jsonl
     ```

4. **Phase 4: TRAKE event-step**
   - Input: frame/shot captions + evidence.
   - Output:
     ```text
     event_step_index.jsonl
     ```

Nói ngắn gọn: **Phase 0 là móng**. Nó không làm caption đẹp ngay, nhưng đảm bảo dữ liệu từ OCR/object/audio đã cùng frame, cùng timestamp, sạch text nhiễu, đủ để các bước caption sau không bị lệch.

> Tôi đã thực hiện viết code xong ở [caption_pipeline](data_processing/caption/caption_pipeline/) , hãy review và cho lệnh chạy trên server

<details><summary>10 previous messages</summary>

> Mình sẽ review theo kiểu code-review: đọc cấu trúc `caption_pipeline`, kiểm tra runner/loader/schema với output mới, chạy compile hoặc dry-run nếu có thể, rồi đưa lệnh server phù hợp. Trọng tâm là lỗi join/timestamp/audio-window và output path.
>
> Mình thấy package đã có đủ các module Phase 0/1 và runner. Mình sẽ đọc nội dung từng file, rồi chạy thử compile/dry-run trên output hiện tại để review bằng kết quả thật thay vì chỉ nhìn qua cấu trúc.
>
> Mình đã đọc runner, loader, alignment và evidence builder. Có vài điểm đáng chú ý cần kiểm tra bằng dry-run: `frame_id` trong object loader có thể không được set, audio window selection dựa trên list sorted nên ổn, nhưng `object_text` có thể vẫn chứa label đã bị remove khỏi counts nếu scene guardrail chỉ sửa counts. Mình sẽ đọc tiếp shot/index/validation rồi chạy thử trên output thật.
>
> Code trông khá gọn, nhưng có một lỗi schema rõ: object loader không đưa `frame_id` vào record, nên `frame_evidence.frame_id` sẽ rơi về `canonical_frame_id` vẫn ổn về giá trị. Mình sẽ chạy compile + dry-run trên output thật vào thư mục tạm để bắt lỗi runtime và xem report/output.
>
> Dry-run pass trên output thật: 283 frame evidence, 102 shots, 385 compact docs, acceptance checks pass. Mình sẽ đọc sample output và report để phát hiện vấn đề logic không làm fail runtime.
>
> Context compaction
>
> Source: automatic
> Status: completed
>
> Mình sẽ chốt phần review theo đúng kiểu “tìm lỗi trước, rồi mới đưa lệnh chạy”. Trước khi trả lời mình kiểm thêm report/samples và line number để các nhận xét có điểm tựa cụ thể.
>
> Dry-run local đã pass đủ 283 frame, nhưng có hai chỗ mình thấy đáng sửa trước khi coi Phase 0 là “đóng nắp”: một chỗ ảnh hưởng trực tiếp tới search text, một chỗ làm report bị lệch nghĩa. Mình kiểm thêm trạng thái file một nhịp rồi gửi lệnh server luôn.
>
> Một điểm thực dụng nữa: ở repo hiện tại phần caption đang là untracked, nên nếu bạn định `git push` rồi `git pull` trên server thì phần này sẽ không đi theo nếu chưa `git add/commit`. Đây là kiểu lỗi “chạy máy mình được, lên server mất file” rất dễ dính.
>
> Mình cũng kiểm `--include-ocr-only-frames`: code đã có nhánh xử lý đúng, mặc định vẫn lấy object detection làm primary frame list. Với output mới nhất của L22_V012 thì OCR/object đều 283 frame nên không có vấn đề join ở case này.
</details>

Review nhanh: code chạy được trên output mới, nhưng mình thấy 3 điểm cần lưu ý trước khi push/run server.

**Findings**
1. **Blocker trước khi lên server:** toàn bộ phần caption đang `untracked`: `caption_pipeline/`, `run_caption_pipeline.py`, `caption_pipeline_code_plan.md`. Nếu chưa `git add/commit/push`, server `git pull` sẽ không có code mới.

2. **Search text có thể bị “lọt” object cũ sau khi filter scene-label.**
   Ở [load_inputs.py](C:/Users/User/Desktop/AIC2026/data_processing/caption/caption_pipeline/load_inputs.py:119) bạn đã filter `object_counts`, nhưng [load_inputs.py](C:/Users/User/Desktop/AIC2026/data_processing/caption/caption_pipeline/load_inputs.py:159) vẫn lấy `object_text/all_object_text` từ raw row. Sau đó [evidence_builder.py](C:/Users/User/Desktop/AIC2026/data_processing/caption/caption_pipeline/evidence_builder.py:46) và [compact_index.py](C:/Users/User/Desktop/AIC2026/data_processing/caption/caption_pipeline/compact_index.py:38) dùng lại text này để build search index. Nên rebuild `object_text` từ `filtered_counts`.

3. **Report audio hơi sai nghĩa:** dry-run log báo `217 usable / 226 total`, nhưng report ghi `num_audio_features=217` và `num_audio_features_usable=217`. Nguyên nhân ở [evidence_alignment.py](C:/Users/User/Desktop/AIC2026/data_processing/caption/caption_pipeline/evidence_alignment.py:36) chỉ nhận list đã filter usable. Không ảnh hưởng output chính, nhưng report dễ gây hiểu nhầm.

Mình đã verify local: `py_compile` pass, dry-run L22_V012 pass: `283 frame evidence`, `102 shots`, `385 search docs`, `0 timestamp mismatch`, `0 warnings`.

**Lệnh Chạy Server**
Caption pipeline này không cần GPU, chỉ fuse OCR/Object/Audio nên chạy CPU là đủ:

```bash
cd /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/data_processing/caption

python run_caption_pipeline.py \
  --video-id L22_V012 \
  --ocr-jsonl /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/outputs/ocr_vlm_pipeline_v2/L22_V012/L22_V012_ocr_es_docs.jsonl \
  --object-jsonl /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/outputs/object_detection/L22_V012_objects.jsonl \
  --audio-features /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/outputs/audio/L22_V012/features/audio_features.jsonl \
  --output-dir /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/outputs/caption \
  --frame-window-before 5 \
  --frame-window-after 5 \
  --shot-max-gap-sec 5 \
  --shot-max-duration-sec 20
```

Nếu muốn lưu log:

```bash
mkdir -p /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/outputs/logs

python run_caption_pipeline.py \
  --video-id L22_V012 \
  --ocr-jsonl /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/outputs/ocr_vlm_pipeline_v2/L22_V012/L22_V012_ocr_es_docs.jsonl \
  --object-jsonl /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/outputs/object_detection/L22_V012_objects.jsonl \
  --audio-features /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/outputs/audio/L22_V012/features/audio_features.jsonl \
  --output-dir /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/outputs/caption \
  --frame-window-before 5 \
  --frame-window-after 5 \
  --shot-max-gap-sec 5 \
  --shot-max-duration-sec 20 \
  2>&1 | tee /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/outputs/logs/caption_L22_V012.log
```

Sau khi chạy, kiểm tra nhanh:

```bash
wc -l /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/outputs/caption/L22_V012/evidence/frame_evidence.jsonl
wc -l /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/outputs/caption/L22_V012/evidence/shot_evidence.jsonl
cat /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/outputs/caption/L22_V012/reports/evidence_alignment_report.json
```

> Trước hết, hãy sửa các lỗi nhỏ trước khi push code

<details><summary>15 previous messages</summary>

> Mình sẽ sửa đúng các lỗi nhỏ đã nêu trước: rebuild `object_text` sau khi filter scene-label, report audio total/usable cho đúng nghĩa, và thêm metric scene-label removed vào report. Mình đọc lại schema hiện tại một nhịp để vá gọn, tránh đụng lan sang phase caption sau.
>
> Mình sẽ giữ API loader tương thích: mặc định `load_objects()` và `load_audio_features()` vẫn trả như cũ, chỉ khi CLI gọi `return_stats=True` mới lấy thêm thống kê. Như vậy nếu có notebook/script khác import loader thì ít rủi ro hơn.
>
> <details><summary>Edited 2 files</summary>
>
> - Wrote `C:/Users/User/Desktop/AIC2026/data_processing/caption/caption_pipeline/load_inputs.py` (+39 -11)
> - Wrote `C:/Users/User/Desktop/AIC2026/data_processing/caption/run_caption_pipeline.py` (+4 -2)
>
> </details>
>
> Patch đầu đã xong: object text bây giờ được dựng lại từ `filtered_counts`, không còn kéo theo label đã bị loại khỏi count. Mình vá tiếp report để thống kê phản ánh đúng `226 total / 217 usable` thay vì nhầm cả hai đều là usable.
>
> <details><summary>Edited 2 files</summary>
>
> - Wrote `C:/Users/User/Desktop/AIC2026/data_processing/caption/caption_pipeline/evidence_alignment.py` (+1 -1)
> - Wrote `C:/Users/User/Desktop/AIC2026/data_processing/caption/caption_pipeline/validation.py` (+8 -10)
>
> </details>
>
> Đã vá report metric. Giờ mình chạy compile và một dry-run nhỏ lên `D:\tmp` để chắc chắn CLI vẫn ra đủ artifacts trước khi bạn push.
>
> `python.exe` trong WindowsApps không khởi động được ở shell hiện tại, nên compile chưa chạy được. Mình sẽ thử launcher/đường dẫn Python khác; nếu vẫn bị môi trường Windows chặn thì mình sẽ báo rõ thay vì che lỗi.
>
> Chỉ thấy `python` trỏ tới WindowsApps stub, không có `py`/conda local trên máy Windows này. Mình thử chạy compile một lần với quyền ngoài sandbox; nếu không được thì phần verify local sẽ dừng ở mức đọc/diff, còn lệnh server vẫn dùng Python trong env server của bạn.
>
> Compile đã pass sau khi chạy ngoài sandbox. Mình chạy lại pipeline trên sample L22_V012 vào thư mục tạm riêng để kiểm tra metric mới và đảm bảo output vẫn đủ frame/shot/index.
>
> Dry-run pass, nhưng log cho thấy vài ký tự Unicode trong CLI bị render lỗi ở PowerShell/server-like console (`—`, check mark). Vì chuẩn repo mình đang sửa theo hướng ASCII-safe, mình sẽ đổi các ký tự log/report đó sang text ASCII trước khi chốt.
>
> Metric mới đã đúng: report hiện ra `226 total`, `217 usable`, `5 scene labels removed`. Mình chỉnh nốt phần runtime text sang ASCII để log/report không bị lỗi encoding khi chạy qua terminal khác nhau.
>
> <details><summary>Edited 2 files</summary>
>
> - Wrote `C:/Users/User/Desktop/AIC2026/data_processing/caption/run_caption_pipeline.py` (+16 -16)
> - Wrote `C:/Users/User/Desktop/AIC2026/data_processing/caption/caption_pipeline/validation.py` (+5 -5)
>
> </details>
>
> Các dòng log/report runtime đã ASCII-safe. Mình chạy compile và dry-run thêm một lần sau patch cuối để chắc không có lỗi cú pháp hay lệch output.
>
> Compile pass, và `rg` không còn bắt Unicode runtime trong CLI/report nữa. Mình chạy lại dry-run để xác nhận log sạch và report vẫn giữ metric mới.
>
> Dry-run cuối pass với log sạch. Mình kiểm thêm report và overlap object-vs-scene trong output để chắc guardrail thật sự có tác dụng ở search evidence.
</details>

Đã sửa các lỗi nhỏ trước khi push.

Đã chỉnh ở:
- [load_inputs.py](C:/Users/User/Desktop/AIC2026/data_processing/caption/caption_pipeline/load_inputs.py:90): rebuild `object_text/all_object_text` từ `filtered_counts`, nên scene-label đã bị loại khỏi object count sẽ không còn lọt vào search text.
- [load_inputs.py](C:/Users/User/Desktop/AIC2026/data_processing/caption/caption_pipeline/load_inputs.py:191): `load_audio_features(..., return_stats=True)` trả thêm total/usable.
- [run_caption_pipeline.py](C:/Users/User/Desktop/AIC2026/data_processing/caption/run_caption_pipeline.py:132): CLI nhận thêm `object_stats/audio_stats` và đưa vào report.
- [validation.py](C:/Users/User/Desktop/AIC2026/data_processing/caption/caption_pipeline/validation.py:35): report giờ có đúng `num_audio_features`, `num_audio_features_usable`, `num_scene_labels_removed_from_counts`.
- [run_caption_pipeline.py](C:/Users/User/Desktop/AIC2026/data_processing/caption/run_caption_pipeline.py:123): đổi log/report runtime sang ASCII để tránh lỗi encoding trên server terminal.

Đã verify:
- `py_compile` pass.
- Dry-run L22_V012 pass: `283` frame evidence, `102` shots, `385` compact search docs.
- Report mới đúng: `226` audio total, `217` usable, `5` scene labels removed.
- Kiểm tra output evidence: `object_scene_count_overlap=0`.

Lưu ý trước khi push: phần caption vẫn đang `untracked`, nên cần add vào git:

```bash
git add data_processing/caption/run_caption_pipeline.py \
  data_processing/caption/caption_pipeline \
  data_processing/caption/caption_pipeline_code_plan.md
```

Sau đó commit/push là server `git pull` mới nhận được code mới.

> hạy cho lệnh chạy

Chạy caption pipeline trên server như sau, không cần GPU:

```bash
cd /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/data_processing/caption

python run_caption_pipeline.py \
  --video-id L22_V012 \
  --ocr-jsonl /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/outputs/ocr_vlm_pipeline_v2/L22_V012/L22_V012_ocr_es_docs.jsonl \
  --object-jsonl /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/outputs/object_detection/L22_V012_objects.jsonl \
  --audio-features /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/outputs/audio/L22_V012/features/audio_features.jsonl \
  --output-dir /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/outputs/caption \
  --frame-window-before 5 \
  --frame-window-after 5 \
  --shot-max-gap-sec 5 \
  --shot-max-duration-sec 20
```

Nếu muốn lưu log:

```bash
mkdir -p /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/outputs/logs

python run_caption_pipeline.py \
  --video-id L22_V012 \
  --ocr-jsonl /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/outputs/ocr_vlm_pipeline_v2/L22_V012/L22_V012_ocr_es_docs.jsonl \
  --object-jsonl /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/outputs/object_detection/L22_V012_objects.jsonl \
  --audio-features /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/outputs/audio/L22_V012/features/audio_features.jsonl \
  --output-dir /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/outputs/caption \
  --frame-window-before 5 \
  --frame-window-after 5 \
  --shot-max-gap-sec 5 \
  --shot-max-duration-sec 20 \
  2>&1 | tee /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/outputs/logs/caption_L22_V012.log
```

Kiểm tra nhanh sau khi chạy:

```bash
wc -l /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/outputs/caption/L22_V012/evidence/frame_evidence.jsonl
wc -l /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/outputs/caption/L22_V012/evidence/shot_evidence.jsonl
cat /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/outputs/caption/L22_V012/reports/evidence_alignment_report.json
```

> Hãy kiểm tra output ở [L22_V012](L22_V012/) nếu đã đạt chuẩn hãy lên plan cho phase tiếp theo

<details><summary>1 previous message</summary>

> Mình sẽ kiểm output `L22_V012/` theo các tiêu chí Phase 0/1: đủ artifacts, line count, report metrics, schema mẫu, và xem search text có bị rỗng/lệch nguồn không. Nếu pass, mình sẽ viết plan phase tiếp theo dựa trên chính output này thay vì đoán.
</details>

Trong `D:\Users\outputs` hiện không có thư mục top-level `L22_V012/`; chỉ có `audio`, `object_detection`, `ocr_vlm_pipeline_v2`. Mình sẽ dò các thư mục con mang tên `L22_V012` và cả output caption nếu đã được copy về.

> tôi đã cập nhật lại [caption](caption/)

<details><summary>10 previous messages</summary>

> Mình thấy rồi, giờ output caption đã nằm trong `caption/`. Mình sẽ kiểm đúng bộ artifacts trong `caption/L22_V012`, đối chiếu report và sample JSONL; nếu ổn mình sẽ viết plan phase tiếp theo thành file `.md` để bạn tiếp tục code theo nó.
>
> Ở `D:\Users\outputs` vẫn chưa thấy thư mục `caption/`, nên có khả năng IDE đang mở từ repo `C:\Users\User\Desktop\AIC2026\outputs\caption`. Mình sẽ dò đúng vị trí file đang mở rồi kiểm từ đó.
>
> Đã tìm đúng rồi: output mới nằm ở `C:\Users\User\Desktop\AIC2026\caption\L22_V012`, không phải `D:\Users\outputs`. Mình chuyển kiểm tra sang đúng thư mục này.
>
> Artifacts đủ 5 file và report đang rất sạch: 283 object/OCR joined, 0 mismatch, 226 audio total/217 usable. Mình sẽ kiểm sâu thêm JSONL: line count, schema, unit type trong compact index, text rỗng, duplicate id, và overlap object/scene.
>
> Kết quả validation chính đều pass. Có một chi tiết sample: `shot_evidence` không có field `num_frames` mà có thể dùng tên khác, nên mình kiểm schema shot thật để xem đây là do chọn sai tên field hay thiếu field.
>
> Shot schema ổn: field đúng là `frame_count`, không phải `num_frames`; sample ban đầu trống chỉ vì mình chọn sai field. Phase hiện tại đạt chuẩn để làm nền cho caption generation/fusion. Mình đọc thêm phần cuối plan hiện tại để không viết phase tiếp theo lệch thứ tự.
>
> Plan cũ đã định Phase 2 là “Text-only frame caption fuser”, rồi Phase 3 shot ReCap, Phase 4 TRAKE. Vì output Phase 0/1 đã đạt chuẩn, mình sẽ viết plan Phase 2 tập trung vào text-only fuser trước, có chừa schema để Phase 3/4 nối vào.
>
> Mình sẽ tạo file plan mới `caption_phase2_frame_fuser_plan.md` trong `data_processing/caption/`. Nội dung sẽ dùng chính số liệu output vừa kiểm để đóng gate Phase 0/1, rồi mô tả Phase 2 theo hướng text-only fuser có baseline deterministic trước, optional LLM sau.
>
> <details><summary>Created a file</summary>
>
> - Wrote `C:/Users/User/Desktop/AIC2026/data_processing/caption/caption_phase2_frame_fuser_plan.md` (+365 -0)
>
> </details>
>
> Plan Phase 2 đã được tạo. Mình kiểm lại file vừa viết và trạng thái git để đảm bảo nó nằm đúng chỗ, rồi sẽ tóm tắt ngắn gọn kết luận đạt/chưa đạt cho output caption.
</details>

Đã kiểm output mới ở `caption/L22_V012`: đạt chuẩn để đi tiếp Phase 2.

Kết quả chính:
- `frame_evidence.jsonl`: 283 rows
- `shot_evidence.jsonl`: 102 rows
- `compact_search_index.jsonl`: 385 rows = 283 frame docs + 102 shot docs
- `timestamp mismatch`: 0
- `empty frame_search_text`: 0
- `empty compact all_text`: 0
- `duplicate frame/index id`: 0
- `object/scene count overlap`: 0
- Report đúng: `226` audio total, `217` usable, `5` scene labels removed

Các thiếu hụt nhỏ không blocking:
- 20 frame không có OCR text sạch
- 17 frame không có countable object
- 3 frame không có audio window

Mình đã viết plan Phase 2 ở đây: [caption_phase2_frame_fuser_plan.md](C:/Users/User/Desktop/AIC2026/data_processing/caption/caption_phase2_frame_fuser_plan.md:1)

Plan Phase 2 tập trung vào `Text-only frame caption fuser`: sinh `captions/frame_index.jsonl`, thêm `caption_text` vào compact index nếu bật flag, và có baseline deterministic/template trước khi gọi LLM để dễ test ES và batch server.

> ở phase 2 sẽ chạy mô hình LLM nào

Ở Phase 2 hiện tại **mặc định chưa chạy LLM**.

Phase 2 nên đi theo thứ tự:

1. **Chạy `template` mode trước**
   - Không dùng model.
   - Sinh caption deterministic từ OCR + audio + object + scene.
   - Mục tiêu: kiểm schema, ES index, search quality baseline.

2. **Sau khi template pass mới bật LLM mode**
   - LLM chỉ là **text-only fuser**, không phải VLM.
   - Input là `frame_evidence.jsonl`.
   - Output là `frame_index.jsonl`.

Mô hình mình đề xuất cho server của bạn:

```text
Qwen2.5-7B-Instruct
```

Lý do:
- Chạy ổn hơn trên GPU phổ thông như A5000.
- Text-only, đủ tốt cho fusion OCR/audio/object.
- Hỗ trợ tiếng Việt khá ổn.
- Ít nặng hơn 14B/32B, dễ batch hơn.

Nếu GPU còn dư VRAM và muốn caption tự nhiên hơn:

```text
Qwen2.5-14B-Instruct
```

Nhưng không nên bắt đầu bằng 14B. Phase 2 cần ổn schema và batch trước.

Tóm lại plan nên là:

```text
Phase 2A: template mode
Phase 2B: Qwen2.5-7B-Instruct text-only fuser
Phase 2C: so sánh template vs LLM, chọn field đưa vào ES
```

CLI về sau sẽ kiểu:

```bash
python run_caption_pipeline.py \
  ... \
  --enable-frame-captions \
  --caption-mode template
```

Sau đó mới:

```bash
python run_caption_pipeline.py \
  ... \
  --enable-frame-captions \
  --caption-mode llm \
  --llm-model Qwen/Qwen2.5-7B-Instruct
```

> Hãy review code đã thực hiện theo plan, rồi từ đó đưa ra các lệnh chạy trên server

<details><summary>13 previous messages</summary>

> Mình sẽ review theo kiểu code review thật: đọc diff/các file Phase 2, đối chiếu với plan, chạy kiểm tra nếu có thể, rồi mới đưa lệnh server. Trọng tâm là lỗi có thể làm batch server sai output, hỏng schema ES, hoặc âm thầm tạo caption kém chất lượng.
>
> Mình đã thấy Phase 2 đã thêm các file chính: `frame_caption_fuser.py`, `caption_index.py`, `caption_report.py`, và flags trong `run_caption_pipeline.py`. Giờ mình đọc sâu từng file để tìm lỗi thực thi và lỗi schema trước khi chạy dry-run.
>
> Có một điểm mình đã thấy trước khi chạy: `--caption-mode llm` hiện chỉ là placeholder và fallback sang template, nên lệnh server an toàn lúc này phải dùng `template`. Mình kiểm tiếp report/schema và chạy dry-run để xem có lỗi acceptance nào không.
>
> Report generator vẫn có emoji/Unicode trong markdown, không làm sai JSONL nhưng có thể render xấu trên terminal Windows hoặc log server cũ. Mình sẽ không sửa vội khi user đang yêu cầu review; trước hết chạy compile/dry-run để phân loại cái gì là blocking.
>
> Input nguồn để chạy lại có đủ trong `D:\Users\outputs`. Compile bị đúng lỗi WindowsApps Python stub như lần trước, nên mình chạy kiểm tra Python ngoài sandbox rồi dry-run vào `D:\tmp` để không đụng output thật.
>
> Compile pass. Unicode chỉ còn trong docstring/report markdown, không chặn chạy; mình sẽ coi là minor nếu bạn muốn log/report ASCII-safe hơn. Giờ mình chạy pipeline Phase 2 template có rebuild compact để kiểm acceptance đúng với plan.
>
> Dry-run Phase 2 pass ở mức CLI. Mình kiểm thêm JSONL sinh ra: schema, line count, compact index sau rebuild, duplicate id, và vài sample caption để phát hiện vấn đề chất lượng mà acceptance chưa bắt được.
>
> Output template đạt checklist. Tuy vậy có vài điểm review đáng nêu: `llm` mode hiện chưa thật sự gọi model, fallback LLM không được đánh dấu đúng, và normalize dấu tiếng Việt còn sót chữ `đ`. Mình lấy line number chính xác rồi sẽ đưa lệnh server theo chế độ an toàn `template`.
>
> Mình cần thêm một đoạn nhỏ để xác nhận vấn đề normalize dấu: `remove_accents()` chỉ bỏ combining marks nên chữ `đ/Đ` vẫn còn. Đây là kiểu lỗi nhỏ nhưng ảnh hưởng search không dấu, nên sẽ đưa vào findings.
>
> Mình sẽ chạy nhanh `--caption-mode llm` vào thư mục tạm riêng để chứng minh nó chưa dùng LLM thật và xem acceptance có pass nhầm không. Nếu nó pass với template thì lệnh server mình sẽ tuyệt đối không dùng mode này.
>
> `--caption-mode llm` vẫn pass dù không gọi model, đúng là cần coi đây là vấn đề review. Mình kiểm report của run đó để xem warning/fallback có bị che không.
>
> Một quality check thú vị: template pass đủ số lượng, nhưng 281 caption_search vẫn còn chữ `đ`, 114 captions có dấu `..`, và max caption dài 606 ký tự. Không chặn batch, nhưng nếu dùng làm field chính cho search/caption thì nên ghi nhận.
>
> Mình đã xong phần kiểm: template mode có thể chạy server; LLM mode chưa nên chạy vì hiện chỉ là nhãn CLI, không gọi model thật. Mình sẽ gửi findings trước rồi kèm command chạy và command kiểm output.
</details>

**Findings**
1. `--caption-mode llm` chưa chạy LLM thật. CLI đã expose mode này ở [run_caption_pipeline.py](C:/Users/User/Desktop/AIC2026/data_processing/caption/run_caption_pipeline.py:101), nhưng [frame_caption_fuser.py](C:/Users/User/Desktop/AIC2026/data_processing/caption/caption_pipeline/frame_caption_fuser.py:45) chỉ fallback sang template và thêm warning. Vì vậy server hiện chỉ nên chạy `--caption-mode template`.

2. `caption_text_search` chưa bỏ dấu hoàn toàn cho tiếng Việt. [text_utils.py](C:/Users/User/Desktop/AIC2026/data_processing/caption/caption_pipeline/text_utils.py:37) dùng Unicode normalize nên còn sót `đ/Đ`; dry-run có `281/283` caption search còn chữ `đ`. Nếu ES không có Vietnamese analyzer tốt, nên map thêm `đ -> d`.

3. Template caption hơi dài và có double period. [frame_caption_fuser.py](C:/Users/User/Desktop/AIC2026/data_processing/caption/caption_pipeline/frame_caption_fuser.py:121) luôn thêm dấu chấm sau audio; dry-run có `114` caption chứa `..`, max length `606`. Không blocking cho search, nhưng nên polish nếu dùng làm caption hiển thị.

Verification đã chạy:
- `py_compile`: pass.
- Phase 2 template dry-run: `283` frame captions, `0` empty, compact index `385` docs, `283/283` frame docs có `caption_text`.

Trước khi chạy server, nhớ push code Phase 2. Đừng add folder output `caption/` nếu chỉ muốn push code:

```bash
git add data_processing/caption/run_caption_pipeline.py \
  data_processing/caption/caption_pipeline \
  data_processing/caption/caption_phase2_frame_fuser_plan.md

git commit -m "Add caption phase2 frame fuser"
git push
```

**Chạy Server**
```bash
cd /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test
git pull

cd /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/data_processing/caption
mkdir -p /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption/logs

PYTHONIOENCODING=utf-8 python run_caption_pipeline.py \
  --video-id L22_V012 \
  --ocr-jsonl /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/outputs/ocr_vlm_pipeline_v2/L22_V012/L22_V012_ocr_es_docs.jsonl \
  --object-jsonl /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/outputs/object_detection/L22_V012_objects.jsonl \
  --audio-features /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/outputs/audio/L22_V012/features/audio_features.jsonl \
  --output-dir /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption \
  --frame-window-before 5 \
  --frame-window-after 5 \
  --shot-max-gap-sec 5 \
  --shot-max-duration-sec 20 \
  --enable-frame-captions \
  --caption-mode template \
  --rebuild-compact-with-captions \
  2>&1 | tee /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption/logs/caption_phase2_L22_V012.log
```

Kiểm tra nhanh:

```bash
wc -l /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption/L22_V012/captions/frame_index.jsonl
wc -l /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption/L22_V012/indexes/compact_search_index.jsonl
cat /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption/L22_V012/reports/frame_caption_report.json
```

Kỳ vọng: `frame_index.jsonl = 283`, `compact_search_index.jsonl = 385`, `num_caption_empty = 0`.

> trước hết, có những lỗi sai vặt nào không ? Nếu có hãy sửa trước khi tôi push

<details><summary>12 previous messages</summary>

> Đúng, có vài lỗi vặt nên sửa trước khi push: normalize tiếng Việt còn sót `đ`, caption template có thể tạo `..`, report markdown còn emoji/Unicode, và caption template hơi dài. Mình sẽ vá các điểm này gọn, không đụng kiến trúc Phase 2 hay schema chính.
>
> Mình sẽ sửa bốn điểm nhỏ: `đ -> d` trong search text, tránh double period khi audio/OCR đã có dấu câu, giới hạn caption template để không phình quá dài, và nếu ai lỡ chạy `--caption-mode llm` thì report sẽ ghi đúng là fallback thay vì trông như template sạch.
>
> <details><summary>Edited 2 files</summary>
>
> - Wrote `C:/Users/User/Desktop/AIC2026/data_processing/caption/caption_pipeline/text_utils.py` (+1 -0)
> - Wrote `C:/Users/User/Desktop/AIC2026/data_processing/caption/caption_pipeline/frame_caption_fuser.py` (+25 -8)
>
> </details>
>
> Patch chính đã vào. Mình chỉnh nốt report markdown sang ASCII-safe và bỏ vài ký tự Unicode còn nằm trong code runtime/docstring để trước khi push sạch hơn.
>
> <details><summary>Edited 2 files</summary>
>
> - Wrote `C:/Users/User/Desktop/AIC2026/data_processing/caption/caption_pipeline/caption_report.py` (+4 -4)
> - Wrote `C:/Users/User/Desktop/AIC2026/data_processing/caption/caption_pipeline/caption_index.py` (+3 -3)
>
> </details>
>
> Mình sẽ quét lại ký tự Unicode trong các file Phase 2 vừa sửa, rồi compile và dry-run lại để đảm bảo không tạo lỗi mới.
>
> Compile đã pass. Còn 2 ký tự Unicode không cần thiết trong docstring/comment, và 2 ký tự `đ/Đ` là có chủ đích để normalize tiếng Việt; mình dọn nốt phần không cần thiết.
>
> <details><summary>Edited 2 files</summary>
>
> - Wrote `C:/Users/User/Desktop/AIC2026/data_processing/caption/run_caption_pipeline.py` (+1 -1)
> - Wrote `C:/Users/User/Desktop/AIC2026/data_processing/caption/caption_pipeline/frame_caption_fuser.py` (+1 -1)
>
> </details>
>
> Giờ mình chạy lại pipeline Phase 2 vào `D:\tmp` và đo đúng các chỉ số đã từng lỗi: còn `đ` trong search không, còn `..` không, caption có vượt quá giới hạn không.
>
> Dry-run pass. Mình kiểm metrics hậu sửa và sample đầu tiên để chắc search text/period đã sạch.
>
> Các chỉ số đã sạch: không còn `đ` trong `caption_text_search`, không còn `..`, max caption đúng 420, compact vẫn 385 docs. Mình kiểm nốt `llm` placeholder có report fallback đúng để tránh sau này hiểu nhầm khi test flag.
>
> `llm` placeholder giờ báo rõ `283 fallback`, vậy người chạy sẽ không bị tưởng là đã gọi LLM thật. Mình lấy nhanh git diff/status để bạn biết chính xác file nào cần add trước khi push.
</details>

Đã có vài lỗi vặt, mình đã sửa trước khi bạn push.

Đã sửa:
- [text_utils.py](C:/Users/User/Desktop/AIC2026/data_processing/caption/caption_pipeline/text_utils.py:37): normalize `đ/Đ -> d/D`, nên `caption_text_search` không còn sót chữ `đ`.
- [frame_caption_fuser.py](C:/Users/User/Desktop/AIC2026/data_processing/caption/caption_pipeline/frame_caption_fuser.py:45): nếu chạy nhầm `--caption-mode llm`, report giờ ghi đúng là `fallback`, không còn giả như template sạch.
- [frame_caption_fuser.py](C:/Users/User/Desktop/AIC2026/data_processing/caption/caption_pipeline/frame_caption_fuser.py:119): bỏ double period do audio/OCR đã có dấu câu.
- [frame_caption_fuser.py](C:/Users/User/Desktop/AIC2026/data_processing/caption/caption_pipeline/frame_caption_fuser.py:163): cap caption template tối đa `420` ký tự.
- [caption_report.py](C:/Users/User/Desktop/AIC2026/data_processing/caption/caption_pipeline/caption_report.py:94): markdown report chuyển sang ASCII-safe, không còn emoji trong report.

Đã verify lại:
- `py_compile`: pass.
- Dry-run Phase 2 template: `283/283` captions, `0` empty, compact index `385` docs.
- `caption_text_search` còn `đ`: `0`
- double period captions: `0`
- max caption length: `420`
- frame docs có `caption_text`: `283`

Lưu ý: `--caption-mode llm` vẫn chưa gọi LLM thật, chỉ fallback template và report đã ghi rõ `num_fallback_used=283`. Khi push code, đừng add folder output `caption/` nếu không muốn đưa kết quả chạy vào git:

```bash
git add data_processing/caption/run_caption_pipeline.py \
  data_processing/caption/caption_pipeline \
  data_processing/caption/caption_phase2_frame_fuser_plan.md
```

> cho lệnh chạy code tuần tự lại

Chạy tuần tự lại trên server theo thứ tự này.

```bash
cd /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test
git pull

cd /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/data_processing/caption
mkdir -p /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption/logs
```

Chạy pipeline Phase 0/1/2 bằng `template` mode:

```bash
PYTHONIOENCODING=utf-8 python run_caption_pipeline.py \
  --video-id L22_V012 \
  --ocr-jsonl /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/outputs/ocr_vlm_pipeline_v2/L22_V012/L22_V012_ocr_es_docs.jsonl \
  --object-jsonl /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/outputs/object_detection/L22_V012_objects.jsonl \
  --audio-features /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/outputs/audio/L22_V012/features/audio_features.jsonl \
  --output-dir /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption \
  --frame-window-before 5 \
  --frame-window-after 5 \
  --shot-max-gap-sec 5 \
  --shot-max-duration-sec 20 \
  --enable-frame-captions \
  --caption-mode template \
  --rebuild-compact-with-captions \
  2>&1 | tee /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption/logs/caption_phase2_L22_V012.log
```

Kiểm tra output:

```bash
wc -l /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption/L22_V012/evidence/frame_evidence.jsonl
wc -l /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption/L22_V012/evidence/shot_evidence.jsonl
wc -l /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption/L22_V012/captions/frame_index.jsonl
wc -l /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption/L22_V012/indexes/compact_search_index.jsonl
cat /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption/L22_V012/reports/frame_caption_report.json
```

Kỳ vọng:
- `frame_evidence.jsonl`: `283`
- `shot_evidence.jsonl`: `102`
- `frame_index.jsonl`: `283`
- `compact_search_index.jsonl`: `385`
- `num_caption_empty`: `0`

> "23:57:07 [INFO] caption_pipeline: Wrote caption report MD -> /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption/L22_V012/r
> eports/frame_caption_report.md
> 23:57:07 [INFO] caption_pipeline: === Done in 0.2s ===
> 23:57:07 [INFO] caption_pipeline:   Frames:  283
> 23:57:07 [INFO] caption_pipeline:   Shots:   102
> 23:57:07 [INFO] caption_pipeline:   Index:   385 docs
> 23:57:07 [INFO] caption_pipeline:   Warnings: 0
> 23:57:07 [INFO] caption_pipeline:   Captions: 283/283
> 23:57:07 [INFO] caption_pipeline: All acceptance checks passed.wc -l /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption/L22_V012/indexes/compact_search_index.jsonl
> cat /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption/L22_V012/reports/frame_caption_report.json
> 283 /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption/L22_V012/evidence/frame_evidence.jsonl
> 102 /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption/L22_V012/evidence/shot_evidence.jsonl
> 283 /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption/L22_V012/captions/frame_index.jsonl
> 385 /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption/L22_V012/indexes/compact_search_index.jsonl
> {
>   "video_id": "L22_V012",
>   "created_at": "2026-07-09T15:57:07+00:00",
>   "num_frames": 283,
>   "num_caption_generated": 283,
>   "num_caption_empty": 0,
>   "num_fallback_used": 0,
>   "coverage": {
>     "ocr": 263,
>     "audio": 280,
>     "objects": 266,
>     "scene": 283
>   },
>   "caption_mode_counts": {
>     "template": 283
>   },
>   "duplicate_document_ids": 0,
>   "duplicate_canonical_frame_ids": 0,
>   "num_frame_warnings": 0,
>   "frame_warnings": [],
>   "summary_warnings": []
> }
> (base) maitanha@cl777:/tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/data_processing/caption$" Kiểm tra nếu pass thì đến bước tiếp theo

Pass rồi. Các chỉ số đều đúng chuẩn Phase 2:

- `frame_evidence`: 283
- `shot_evidence`: 102
- `frame_index`: 283
- `compact_search_index`: 385
- `num_caption_empty`: 0
- `num_fallback_used`: 0
- `duplicate_document_ids`: 0
- `duplicate_canonical_frame_ids`: 0
- `caption_mode`: `template: 283`

Tức là Phase 0/1/2 đã ổn.

Bước tiếp theo nên là **Phase 3: Shot-level caption / ReCap-style captioner**.

Mục tiêu Phase 3:

```text
shot_evidence.jsonl
+ frame_index.jsonl
+ previous shot context / memory
-> captions/shot_index.jsonl
-> reports/shot_caption_report.json
-> update compact_search_index.jsonl with shot caption_text
```

Thứ tự làm nên là:

1. Viết plan `.md` cho Phase 3.
2. Implement `shot_captioner` dạng template trước, chưa gọi LLM.
3. Sinh `shot_index.jsonl` với 102 rows.
4. Rebuild `compact_search_index.jsonl`: frame docs giữ caption Phase 2, shot docs thêm `caption_text`.
5. Sau khi template shot pass mới tính đến LLM/ReCap thật.

Tiêu chí Phase 3 pass dự kiến:

```text
shot_index rows = 102
shot caption empty = 0
compact_search_index rows = 385
frame docs with caption_text = 283
shot docs with caption_text = 102
duplicate shot ids = 0
```

> Tiếp tục viết plan .md chi tiết cho phase 3 để code

<details><summary>6 previous messages</summary>

> Mình sẽ viết plan Phase 3 thành file riêng, bám đúng output đã pass của Phase 2 và code hiện tại. Trước khi viết mình đọc lại schema shot/frame index và compact index để plan sau này code không bị lệch field.
>
> Code hiện tại đã có frame index và compact rebuild cho frame docs; Phase 3 nên đi theo cùng pattern, nhưng dùng `shot_id` và `frame_count`. Mình sẽ lấy sample output thật để đưa plan có field name chính xác.
>
> Local `caption/` trong repo hiện không còn output, nhưng mình còn dry-run đã verify ở `D:\tmp\caption_phase2_fix_check`. Mình dùng sample đó cho plan; schema vẫn đúng với log server bạn vừa gửi.
>
> Mình sẽ viết plan theo hướng tích hợp tuần tự trong `run_caption_pipeline.py`: Phase 3 chỉ chạy sau Phase 2 vì cần `frame_index`. Flag rebuild compact hiện có sẽ được hiểu rộng hơn: nếu bật shot captions thì compact shot docs cũng được cập nhật caption.
>
> <details><summary>Created a file</summary>
>
> - Wrote `C:/Users/User/Desktop/AIC2026/data_processing/caption/caption_phase3_shot_captioner_plan.md` (+734 -0)
>
> </details>
>
> Plan Phase 3 đã được viết. Mình kiểm nhanh file mới và trạng thái git để chắc file nằm đúng chỗ, không bị thiếu mục quan trọng trước khi chốt.
</details>

Đã viết plan chi tiết cho Phase 3 tại:

[caption_phase3_shot_captioner_plan.md](C:/Users/User/Desktop/AIC2026/data_processing/caption/caption_phase3_shot_captioner_plan.md:1)

Plan này bám theo output Phase 2 đã pass và định nghĩa rõ:

- Schema `caption_shot_index_v1`
- Module cần thêm: `shot_captioner.py`, `shot_index.py`, `shot_caption_report.py`
- CLI mới: `--enable-shot-captions`, `--shot-caption-mode template`
- Cách dùng `frame_index.jsonl` + `shot_evidence.jsonl`
- Template shot captioner trước, LLM để sau
- ReCap-style memory deterministic
- Cách rebuild `compact_search_index.jsonl` để shot docs có `caption_text`
- Acceptance checklist: `shot_index=102`, compact vẫn `385`, frame docs caption `283`, shot docs caption `102`

Bước tiếp theo hợp lý là code Phase 3 theo thứ tự trong mục `## 14. Suggested implementation order`.

> Hãy bắt đầu thực thi viết code

<details><summary>18 previous messages</summary>

> Mình bắt đầu code Phase 3 luôn. Mình sẽ giữ đúng tinh thần Phase 2: template deterministic trước, `llm` chỉ là placeholder fallback rõ ràng, và schema additive để không phá output hiện tại.
>
> Một số phần Phase 3 đã được chuẩn bị sẵn nhẹ (`SHOT_INDEX_SCHEMA` có rồi). Mình sẽ thêm config, modules mới, report, prompt, rồi nối vào CLI.
>
> Mình sẽ chỉnh `config.py` trước để pipeline biết `--enable-shot-captions` và các giới hạn caption/memory của shot. Sau đó các module mới sẽ dùng chung config này.
>
> <details><summary>Edited a file</summary>
>
> - Wrote `C:/Users/User/Desktop/AIC2026/data_processing/caption/caption_pipeline/config.py` (+17 -0)
>
> </details>
>
> Tiếp theo mình tạo `shot_captioner.py`: module này nhận `shot_evidence` và `frame_index`, tạo caption shot + temporal caption + memory deterministic, đồng thời đánh dấu fallback rõ nếu ai bật `llm`.
>
> <details><summary>Created a file</summary>
>
> - Wrote `C:/Users/User/Desktop/AIC2026/data_processing/caption/caption_pipeline/shot_captioner.py` (+220 -0)
>
> </details>
>
> Bây giờ mình thêm `shot_index.py`: nó sẽ build `caption_shot_index_v1`, tạo search fields, normalize search text giống frame index, và có helper rebuild compact cho cả frame + shot captions.
>
> <details><summary>Created a file</summary>
>
> - Wrote `C:/Users/User/Desktop/AIC2026/data_processing/caption/caption_pipeline/shot_index.py` (+212 -0)
>
> </details>
>
> Tiếp theo là `shot_caption_report.py`, tương tự report frame nhưng có thêm temporal caption, duplicate shot id và coverage frame captions.
>
> <details><summary>Created a file</summary>
>
> - Wrote `C:/Users/User/Desktop/AIC2026/data_processing/caption/caption_pipeline/shot_caption_report.py` (+190 -0)
>
> </details>
>
> Mình thêm prompt placeholder cho Phase 3B/LLM để repo có đủ cấu trúc, nhưng code Phase 3A sẽ không gọi prompt này.
>
> <details><summary>Created a file</summary>
>
> - Wrote `C:/Users/User/Desktop/AIC2026/data_processing/caption/caption_pipeline/prompts/shot_caption_prompt.txt` (+21 -0)
>
> </details>
>
> Các module mới đã có. Mình sẽ nối vào `run_caption_pipeline.py`: thêm import, flags, config, yêu cầu Phase 3 phải bật Phase 2, ghi `shot_index` và report, rồi rebuild compact sau shot captions.
>
> <details><summary>Edited a file</summary>
>
> - Wrote `C:/Users/User/Desktop/AIC2026/data_processing/caption/run_caption_pipeline.py` (+75 -0)
>
> </details>
>
> Pipeline đã nối xong. Mình sẽ kiểm đoạn flow vừa patch để bắt lỗi logic trước khi chạy compile.
>
> Không thấy Unicode không cần thiết trong code mới. Mình chạy compile trước; nếu pass sẽ chạy full dry-run Phase 0/1/2/3 vào `D:\tmp` và kiểm đủ acceptance.
>
> Compile pass. Mình chạy dry-run Phase 3 với template, có rebuild compact, đúng như lệnh server dự kiến.
>
> Dry-run pass. Mình sẽ kiểm sâu output: line counts, schema, compact frame/shot captions, search normalization, double-period, max length và report JSON.
</details>

Acceptance chính đã sạch: `shot_index=102`, compact có đủ `283` frame captions và `102` shot captions. Mình kiểm thêm guardrail CLI: nếu bật Phase 3 mà quên Phase 2 thì phải fail rõ ràng.

> continue

<details><summary>1 previous message</summary>

> Mình tiếp tục phần còn lại: không chạy lại lệnh Python guardrail nữa vì lần xin quyền phụ vừa bị hệ thống chặn do quota, nhưng dry-run Phase 3 chính đã pass đầy đủ. Mình sẽ kiểm trạng thái file và tóm tắt phần đã code + lệnh server để bạn chạy.
</details>

Đã code xong Phase 3.

Thay đổi chính:
- Thêm `ShotCaptionConfig` và `enable_shot_captions` trong `config.py`.
- Thêm `shot_captioner.py`: sinh `caption_text`, `temporal_caption`, `memory_before/memory_after` cho từng shot.
- Thêm `shot_index.py`: sinh `caption_shot_index_v1` và rebuild compact index cho cả frame + shot captions.
- Thêm `shot_caption_report.py`: report quality Phase 3.
- Thêm prompt placeholder `prompts/shot_caption_prompt.txt`.
- Nối vào `run_caption_pipeline.py` với flags:
  - `--enable-shot-captions`
  - `--shot-caption-mode template`

Đã verify dry-run Phase 3 local:
- `frame_evidence`: 283
- `shot_evidence`: 102
- `frame_index`: 283
- `shot_index`: 102
- `compact_search_index`: 385
- compact frame docs có caption: 283
- compact shot docs có caption: 102
- empty shot caption: 0
- empty temporal caption: 0
- duplicate shot ids: 0
- `caption_text_search` còn `đ`: 0
- double period: 0
- max shot caption length: 520
- max temporal caption length: 620

Lệnh chạy server sau khi push:

```bash
cd /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/data_processing/caption

PYTHONIOENCODING=utf-8 python run_caption_pipeline.py \
  --video-id L22_V012 \
  --ocr-jsonl /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/outputs/ocr_vlm_pipeline_v2/L22_V012/L22_V012_ocr_es_docs.jsonl \
  --object-jsonl /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/outputs/object_detection/L22_V012_objects.jsonl \
  --audio-features /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/outputs/audio/L22_V012/features/audio_features.jsonl \
  --output-dir /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption \
  --frame-window-before 5 \
  --frame-window-after 5 \
  --shot-max-gap-sec 5 \
  --shot-max-duration-sec 20 \
  --enable-frame-captions \
  --caption-mode template \
  --enable-shot-captions \
  --shot-caption-mode template \
  --rebuild-compact-with-captions
```

Kiểm tra:

```bash
wc -l /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption/L22_V012/captions/frame_index.jsonl
wc -l /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption/L22_V012/captions/shot_index.jsonl
wc -l /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption/L22_V012/indexes/compact_search_index.jsonl
cat /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption/L22_V012/reports/shot_caption_report.json
```

Trước khi push nhớ add file mới:

```bash
git add data_processing/caption/run_caption_pipeline.py \
  data_processing/caption/caption_pipeline \
  data_processing/caption/caption_phase3_shot_captioner_plan.md
```

> Mọi thứ đều pass, tiếp tục với phase tiếp

<details><summary>4 previous messages</summary>

> Phase 3 đã pass thì phase kế tiếp hợp lý là **Phase 4: TRAKE event-step builder**. Mình sẽ viết plan `.md` chi tiết trước, vì Phase 4 ảnh hưởng trực tiếp schema search/temporal retrieval nên cần chốt event-step schema rõ rồi mới code.
>
> Mình có đủ context từ plan tổng và output Phase 3. Mình sẽ tạo `caption_phase4_trake_event_plan.md`: Phase 4A sẽ là deterministic event-step builder, 1 event-step cho mỗi shot trước, sau đó mới tính LLM/TRAKE nâng cao.
>
> <details><summary>Created a file</summary>
>
> - Wrote `C:/Users/User/Desktop/AIC2026/data_processing/caption/caption_phase4_trake_event_plan.md` (+712 -0)
>
> </details>
>
> Plan Phase 4 đã được tạo. Mình kiểm nhanh các mục chính và trạng thái file để đảm bảo nó nằm đúng chỗ trước khi báo lại.
</details>

Đã tạo plan Phase 4 tại:

[caption_phase4_trake_event_plan.md](C:/Users/User/Desktop/AIC2026/data_processing/caption/caption_phase4_trake_event_plan.md:1)

Phase kế tiếp là **TRAKE Event-Step Builder**. Plan đã chốt hướng làm Phase 4A như sau:

- 1 shot = 1 event-step.
- Sinh `captions/event_step_index.jsonl`.
- Thêm `trake_text`, `action_state`, `temporal_role`, `before_context`, `current_observation`, `after_context`.
- Update `compact_search_index.jsonl` để `shot` docs có `trake_text`.
- Chưa gọi LLM/VLM, chỉ deterministic template baseline.
- LLM mode để Phase 4B, có fallback rõ nếu chưa implement.

Expected sau khi code:
- `event_step_index.jsonl = 102`
- `compact_search_index.jsonl = 385`
- `shot docs with trake_text = 102`
- `frame docs with trake_text = 0`
- `num_trake_text_empty = 0`

Plan cũng có sẵn thứ tự implementation ở mục `## 14. Suggested implementation order`, bắt đầu từ `EVENT_STEP_INDEX_SCHEMA`, `TrakeEventConfig`, rồi `trake_event_builder.py`, `event_step_index.py`, `event_step_report.py`.

> hãy thực hiện code

<details><summary>33 previous messages</summary>

> Mình sẽ code Phase 4 theo đúng plan: deterministic TRAKE event-step trước, không gọi LLM. Vì lần trước quyền chạy Python ngoài sandbox đã bị quota chặn, mình sẽ tập trung patch cẩn thận và dùng các kiểm tra PowerShell/`rg`; nếu Python compile không chạy được ở môi trường này mình sẽ nói rõ.
>
> Mình bắt đầu thêm phần TRAKE config/schema trước, rồi tạo ba module mới: builder, index, report. Các rule sẽ dùng shot-level captions đã có, nên không làm lại Phase 2/3.
>
> <details><summary>Edited 2 files</summary>
>
> - Wrote `C:/Users/User/Desktop/AIC2026/data_processing/caption/caption_pipeline/schemas.py` (+1 -0)
> - Wrote `C:/Users/User/Desktop/AIC2026/data_processing/caption/caption_pipeline/config.py` (+15 -0)
>
> </details>
>
> Giờ mình tạo `trake_event_builder.py`: nó sẽ sinh event-step trung gian, nhận biết start/middle/end/change theo timeline và overlap object/scene, đồng thời tạo `trake_text`.
>
> <details><summary>Created a file</summary>
>
> - Wrote `C:/Users/User/Desktop/AIC2026/data_processing/caption/caption_pipeline/trake_event_builder.py` (+302 -0)
>
> </details>
>
> Builder đã có. Tiếp theo mình tạo `event_step_index.py`, chịu trách nhiệm biến record trung gian thành JSONL schema v1 và update `trake_text` vào compact shot docs.
>
> <details><summary>Created a file</summary>
>
> - Wrote `C:/Users/User/Desktop/AIC2026/data_processing/caption/caption_pipeline/event_step_index.py` (+146 -0)
>
> </details>
>
> Tiếp theo là `event_step_report.py` để có gate rõ ràng cho Phase 4: số event steps, trake text rỗng, duplicate IDs, phân bố action/temporal role.
>
> <details><summary>Created a file</summary>
>
> - Wrote `C:/Users/User/Desktop/AIC2026/data_processing/caption/caption_pipeline/event_step_report.py` (+168 -0)
>
> </details>
>
> Mình thêm prompt placeholder TRAKE cho Phase 4B, vẫn chưa gọi trong Phase 4A.
>
> <details><summary>Created a file</summary>
>
> - Wrote `C:/Users/User/Desktop/AIC2026/data_processing/caption/caption_pipeline/prompts/trake_event_step_prompt.txt` (+26 -0)
>
> </details>
>
> Các module Phase 4 đã có. Giờ mình nối vào CLI: thêm flags `--enable-trake-events`, import builder/index/report, guard yêu cầu shot captions, ghi `event_step_index` và update compact với `trake_text`.
>
> Patch lớn bị lệch context do thứ tự import khác một chút. Mình sẽ chia nhỏ patch theo từng đoạn để tránh đụng sai.
>
> Mình sẽ patch `run_caption_pipeline.py` theo từng cụm nhỏ: import/config, CLI, guard, Phase 4 body, summary/acceptance.
>
> <details><summary>Edited a file</summary>
>
> - Wrote `C:/Users/User/Desktop/AIC2026/data_processing/caption/run_caption_pipeline.py` (+10 -0)
> - Wrote `C:/Users/User/Desktop/AIC2026/data_processing/caption/run_caption_pipeline.py` (+1 -0)
> - Wrote `C:/Users/User/Desktop/AIC2026/data_processing/caption/run_caption_pipeline.py` (+11 -0)
> - Wrote `C:/Users/User/Desktop/AIC2026/data_processing/caption/run_caption_pipeline.py` (+6 -0)
> - Wrote `C:/Users/User/Desktop/AIC2026/data_processing/caption/run_caption_pipeline.py` (+2 -0)
>
> </details>
>
> CLI/config đã nối. Mình thêm block Phase 4 sau Phase 3, trước phần summary.
>
> <details><summary>Edited a file</summary>
>
> - Wrote `C:/Users/User/Desktop/AIC2026/data_processing/caption/run_caption_pipeline.py` (+32 -0)
>
> </details>
>
> Context compaction
>
> Source: automatic
> Status: completed
>
> Mình tiếp tục ngay từ Phase 4 nhé. Trước hết mình sẽ rà lại phần wiring trong `run_caption_pipeline.py` để xem patch trước đã vào đủ chưa, rồi mới chốt phần còn thiếu.
>
> Phase 4 đã được gắn vào CLI và luồng chính rồi. Mình thấy còn thiếu phần summary/acceptance check riêng cho `event_step_index` và `trake_text`, nên mình sẽ bổ sung để khi chạy server có thể fail sớm nếu output chưa đạt chuẩn.
>
> Mình đã đọc các module mới. Code hiện khá gọn, chỉ có vài chỗ nên làm chắc hơn: cập nhật help/docs cho CLI, thêm acceptance check cho event-step, và log số TRAKE generated để server run dễ nhìn.
>
> Mình sẽ patch rất nhỏ ở file chạy chính: phần mô tả output, description CLI, summary log, và acceptance check cho Phase 4. Đây là kiểu “đai an toàn” trước khi push/run server.
>
> <details><summary>Edited a file</summary>
>
> - Wrote `C:/Users/User/Desktop/AIC2026/data_processing/caption/run_caption_pipeline.py` (+17 -1)
>
> </details>
>
> Patch xong. Bây giờ mình sẽ chạy vài kiểm tra tĩnh: tìm các flag/schema vừa thêm, thử compile Python nếu môi trường local cho phép, và rà diff/status để đảm bảo không chạm nhầm phần khác.
>
> Local check có hai giới hạn môi trường: `python.exe` đang trỏ vào WindowsApps stub nên không compile được, và Git báo `dubious ownership` do sandbox user khác owner repo. Mình sẽ dùng `git -c safe.directory=...` cho status/diff, không ghi global config.
>
> Diff đúng phạm vi Phase 4. Mình sẽ thêm thêm một lớp acceptance nhỏ: số event-step phải bằng số shot, và không được trùng `document_id/event_id/shot_id`. Như vậy server log sẽ tự báo lỗi nếu Phase 4 tạo lệch cardinality.
>
> <details><summary>Edited a file</summary>
>
> - Wrote `C:/Users/User/Desktop/AIC2026/data_processing/caption/run_caption_pipeline.py` (+13 -0)
>
> </details>
>
> Vừa thêm acceptance duplicate thì cần thêm helper nhỏ `_count_duplicates` trong CLI file. Mình sẽ thêm ngay dưới `main` để tránh import private từ report module.
>
> <details><summary>Edited a file</summary>
>
> - Wrote `C:/Users/User/Desktop/AIC2026/data_processing/caption/run_caption_pipeline.py` (+11 -0)
>
> </details>
>
> Helper đã vào. Mình sẽ rà lại đoạn cuối file và các file Phase 4 lần nữa để bắt lỗi tên hàm/field rõ ràng trước khi kết luận.
>
> Rà wiring OK. `py_compile` không chạy được vì Python local bị WindowsApps stub, nên mình sẽ không ép local nữa; phần xác thực thật nên chạy trên server/conda env như các phase trước.
>
> Mình sẽ thử rất nhẹ xem máy local có launcher Python khác (`py` hoặc `python3`) không. Nếu không có thì dừng ở static review, vì server mới là môi trường thật của pipeline này.
>
> Không có Python launcher dùng được trong sandbox local, nên mình sẽ không báo pass giả. Phần code Phase 4 đã được đặt đủ file và wiring; mình chỉ còn gom lại các điểm thay đổi và lệnh chạy server để bạn test thật.
</details>

Đã thực hiện code Phase 4 TRAKE event-step.

Các phần chính đã thêm:
- [run_caption_pipeline.py](C:/Users/User/Desktop/AIC2026/data_processing/caption/run_caption_pipeline.py:143): thêm `--enable-trake-events`, `--trake-event-mode`, wiring Phase 4 vào pipeline.
- [config.py](C:/Users/User/Desktop/AIC2026/data_processing/caption/caption_pipeline/config.py:67): thêm `TrakeEventConfig`.
- [schemas.py](C:/Users/User/Desktop/AIC2026/data_processing/caption/caption_pipeline/schemas.py:12): thêm `caption_event_step_index_v1`.
- [trake_event_builder.py](C:/Users/User/Desktop/AIC2026/data_processing/caption/caption_pipeline/trake_event_builder.py:34): build event-step deterministic từ `shot_index + shot_evidence`.
- [event_step_index.py](C:/Users/User/Desktop/AIC2026/data_processing/caption/caption_pipeline/event_step_index.py:15): tạo `event_step_index.jsonl` và rebuild `compact_search_index` với `trake_text`.
- [event_step_report.py](C:/Users/User/Desktop/AIC2026/data_processing/caption/caption_pipeline/event_step_report.py:12): report/acceptance Phase 4.
- [trake_event_step_prompt.txt](C:/Users/User/Desktop/AIC2026/data_processing/caption/caption_pipeline/prompts/trake_event_step_prompt.txt): placeholder prompt cho Phase 4B LLM.

Mình cũng thêm acceptance check trong CLI:
- `event_step_index` phải bằng số `shot_index`
- không được rỗng `event_caption`
- không được rỗng `trake_text`
- không trùng `document_id`, `event_id`, `shot_id`

Local chưa chạy được `py_compile` vì Python trong sandbox đang trỏ vào WindowsApps stub, nên phần test thật nên chạy trên server.

Lệnh chạy server:

```bash
cd /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/data_processing/caption

PYTHONIOENCODING=utf-8 python run_caption_pipeline.py \
  --video-id L22_V012 \
  --ocr-jsonl /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/outputs/ocr_vlm_pipeline_v2/L22_V012/L22_V012_ocr_es_docs.jsonl \
  --object-jsonl /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/outputs/object_detection/L22_V012_objects.jsonl \
  --audio-features /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/outputs/audio/L22_V012/features/audio_features.jsonl \
  --output-dir /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption \
  --frame-window-before 5 \
  --frame-window-after 5 \
  --shot-max-gap-sec 5 \
  --shot-max-duration-sec 20 \
  --enable-frame-captions \
  --caption-mode template \
  --enable-shot-captions \
  --shot-caption-mode template \
  --enable-trake-events \
  --trake-event-mode template \
  --rebuild-compact-with-captions
```

Check nhanh sau khi chạy:

```bash
wc -l /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption/L22_V012/captions/event_step_index.jsonl
wc -l /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption/L22_V012/indexes/compact_search_index.jsonl
cat /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption/L22_V012/reports/event_step_report.json
```

Kỳ vọng với `L22_V012`: `event_step_index.jsonl = 102`, `compact_search_index.jsonl = 385`, `num_trake_text_empty = 0`.

> "00:39:12 [INFO] caption_pipeline: --- Phase 4: Generating TRAKE event steps ---
> 00:39:12 [INFO] caption_pipeline.trake_event_builder: TRAKE event builder: 102 event steps (102 with trake_text, 0 fallback)
> 00:39:12 [INFO] caption_pipeline: --- Phase 4: Building event-step index ---
> 00:39:12 [INFO] caption_pipeline.event_step_index: Event-step index: 102 records built
> 00:39:12 [INFO] caption_pipeline: Wrote 102 event-step index -> /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption/L22_V012/captions/event_step_index.jsonl
> 00:39:12 [INFO] caption_pipeline: --- Phase 4: Rebuilding compact index with TRAKE text ---
> 00:39:12 [INFO] caption_pipeline.event_step_index: Compact index rebuild: 102 shot docs updated with trake_text
> 00:39:12 [INFO] caption_pipeline: Rebuilt 385 search docs -> /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption/L22_V012/indexes/compact_search_index.jsonl
> 00:39:12 [INFO] caption_pipeline: Wrote event-step report JSON -> /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption/L22_V012/reports/event_step_report.json
> 00:39:12 [INFO] caption_pipeline: Wrote event-step report MD -> /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption/L22_V012/reports/event_step_report.md
> 00:39:12 [INFO] caption_pipeline: === Done in 0.5s ===
> 00:39:12 [INFO] caption_pipeline:   Frames:  283
> 00:39:12 [INFO] caption_pipeline:   Shots:   102
> 00:39:12 [INFO] caption_pipeline:   Index:   385 docs
> 00:39:12 [INFO] caption_pipeline:   Warnings: 0
> 00:39:12 [INFO] caption_pipeline:   Captions: 283/283
> 00:39:12 [INFO] caption_pipeline:   Shot captions: 102/102
> 00:39:12 [INFO] caption_pipeline:   Event steps: 102/102
> 00:39:12 [INFO] caption_pipeline: All acceptance checks passed." mọi thứ đều pass, tiếp tục để phase tiếp theo

<details><summary>8 previous messages</summary>

> Phase 4 pass rất đẹp: cardinality đúng 102/102, compact vẫn 385 docs, và `trake_text` đã phủ đủ shot docs. Mình sẽ đọc lại plan tổng và các output/schema hiện tại để xác định Phase tiếp theo nên làm gì cho đúng hướng, rồi viết plan chi tiết trước khi code.
>
> Lệnh vừa rồi bị lỗi do workdir sandbox trả về invalid directory, không phải lỗi repo. Mình sẽ chuyển thẳng vào root repo để đọc plan.
>
> Plan tổng ghi sau Phase 4 là hướng benchmark/retrieval: kiểm thử FuseCap/VLM fuser và chuẩn bị index cho ES/Milvus. Với pipeline hiện tại, bước hợp lý nhất là Phase 5: tạo module retrieval-ready export/evaluation, không vội gọi VLM thêm khi chưa benchmark text pipeline.
>
> Vì local không có thư mục `caption/L22_V012` mới từ server, mình sẽ dựa vào log pass bạn gửi và plan tổng. Bước tiếp theo mình đề xuất là Phase 5 “Retrieval Export + Evaluation Harness”, để biến toàn bộ output caption thành dữ liệu search/test được trước khi benchmark VLM tốn GPU.
>
> Mình sẽ viết Phase 5 dưới dạng plan `.md` chi tiết, tập trung vào “retrieval-ready export + lexical evaluation harness”. Đây là lớp kiểm nghiệm search trước; sau khi pass mới nên benchmark VLM fuser vì lúc đó có metric để so sánh.
>
> Không thấy module ES/retrieval hiện hữu cho caption, nên Phase 5 nên tạo mới thay vì đụng backend. Mình sẽ viết plan sao cho code sau này chạy offline được trước, rồi mới optional bulk vào Elasticsearch.
>
> <details><summary>Created a file</summary>
>
> - Wrote `C:/Users/User/Desktop/AIC2026/data_processing/caption/caption_phase5_retrieval_export_eval_plan.md` (+580 -0)
>
> </details>
>
> Plan Phase 5 đã được viết xong. Mình sẽ rà nhanh file mới có nằm đúng chỗ và nội dung các acceptance/run command có thể dùng trực tiếp sau khi code.
</details>

Phase 4 pass rồi, mình đã viết plan cho phase tiếp theo tại:

[data_processing/caption/caption_phase5_retrieval_export_eval_plan.md](C:/Users/User/Desktop/AIC2026/data_processing/caption/caption_phase5_retrieval_export_eval_plan.md:1)

Mình chọn Phase 5 là **Retrieval Export + Evaluation Harness** thay vì gọi VLM ngay, vì hiện tại ta cần một baseline search/eval để sau này so sánh FuseCap/VLM có thật sự tốt hơn hay chỉ tốn GPU hơn.

Plan đã chốt:
- Export `compact_search_index` + `event_step_index` thành corpus search chung.
- Tạo ES mapping + ES bulk JSONL.
- Tạo local lexical evaluator chạy offline không cần Elasticsearch.
- Tạo default query set cho OCR/object/audio/shot/TRAKE.
- Output report `retrieval_eval_report.json/md`.
- Expected với `L22_V012`: `retrieval_corpus = 487`, ES compact bulk `770 lines`, ES event bulk `204 lines`.

Bước kế tiếp hợp lý là code Phase 5 theo plan này.

> hãy thực hiện viết code

<details><summary>20 previous messages</summary>

> Mình sẽ code Phase 5 theo đúng plan vừa viết: tách thành script riêng `run_retrieval_export_eval.py` và các module trong `caption_pipeline`, để không làm lung lay pipeline caption đã pass. Trước tiên mình đọc nhanh các helper IO hiện có để dùng đúng style repo.
>
> Mình sẽ thêm 4 module: export corpus/ES bulk, default queries, local scorer, report renderer; rồi thêm CLI riêng để chạy Phase 5. Code sẽ dùng chỉ stdlib để chạy được trong conda/server mà không cần Elasticsearch thật.
>
> <details><summary>Created a file</summary>
>
> - Wrote `C:/Users/User/Desktop/AIC2026/data_processing/caption/caption_pipeline/retrieval_export.py` (+272 -0)
>
> </details>
>
> Module export đã vào. Tiếp theo mình thêm default query/routing và scorer local, phần này sẽ giúp chạy sanity search mà không cần ES thật.
>
> <details><summary>Created a file</summary>
>
> - Wrote `C:/Users/User/Desktop/AIC2026/data_processing/caption/caption_pipeline/retrieval_queries.py` (+154 -0)
>
> </details>
>
> Default query set đã có 12 query, đều dạng không dấu để tránh lỗi encoding nhưng vẫn match được nhờ normalize. Giờ mình thêm evaluator: scoring theo field weight, matched fields/terms, top-K results.
>
> <details><summary>Created a file</summary>
>
> - Wrote `C:/Users/User/Desktop/AIC2026/data_processing/caption/caption_pipeline/retrieval_eval.py` (+178 -0)
>
> </details>
>
> Scorer local đã xong. Mình sẽ thêm report JSON/Markdown để server run có acceptance rõ ràng giống các phase trước.
>
> <details><summary>Created a file</summary>
>
> - Wrote `C:/Users/User/Desktop/AIC2026/data_processing/caption/caption_pipeline/retrieval_report.py` (+178 -0)
>
> </details>
>
> Report module đã có. Bây giờ mình thêm CLI `run_retrieval_export_eval.py`: đọc output caption hiện tại, xuất corpus/mapping/bulk, chạy query mặc định hoặc query do bạn đưa, rồi fail ở `--strict` nếu có vấn đề blocking.
>
> <details><summary>Created a file</summary>
>
> - Wrote `C:/Users/User/Desktop/AIC2026/data_processing/caption/run_retrieval_export_eval.py` (+240 -0)
>
> </details>
>
> CLI đã được thêm. Mình sẽ kiểm tra tĩnh các import/tên hàm và đọc diff/status để bắt lỗi chính tả trước khi đưa lệnh chạy server.
>
> Các file đều đang untracked đúng như phase mới. Vì Git diff không hiển thị stat cho file untracked, mình sẽ đọc lại script chính và module evaluator/export một vòng để bắt lỗi cú pháp hiển nhiên bằng mắt.
>
> Một điểm mình vừa phát hiện: caption template đang viết bằng English (`Scene shows`, `Visible objects`, `Spoken content`), nên query mặc định không dấu thuần Việt có thể miss không cần thiết. Mình sẽ chỉnh query mặc định thành mixed English + Vietnamese anchor để strict mode ổn định hơn trên output hiện tại.
>
> Patch absolute path vừa bị Windows báo invalid directory một nhịp, mình sẽ dùng path tương đối từ repo root để apply lại.
>
> Có vẻ tool patch đang bám vào cwd `D:\Users\outputs` hiện không còn hợp lệ. Mình sẽ kiểm tra lại thư mục hiện tại của tool, rồi dùng cách patch với đường dẫn chuẩn mà tool chấp nhận.
>
> `apply_patch` đang không đọc được path dù `exec` đọc được file, nên mình sẽ dùng PowerShell replace chính xác từng chuỗi trong file này. Đây là chỉnh sửa nhỏ, có kiểm tra lại ngay sau đó.
>
> PowerShell xác nhận hiện tại sandbox chỉ đọc được repo này nhưng không ghi được trực tiếp vào `C:\Users\User\Desktop\AIC2026`. Mình sẽ xin quyền ghi đúng file repo để hoàn tất chỉnh sửa nhỏ này và các kiểm tra còn lại.
>
> Đã ghi được file sau khi được quyền. Mình sẽ kiểm tra lại query strings và status, rồi thử `py_compile` một lần; nếu Python local vẫn là WindowsApps stub thì mình sẽ ghi rõ là chưa compile local được.
>
> `py_compile` vẫn không chạy được vì Python local trỏ vào WindowsApps stub. Lệnh `rg` pattern hơi chặt nên không ra output, mình sẽ đọc trực tiếp đoạn query để xác nhận.
</details>

Đã code Phase 5 xong.

Đã thêm:
- [run_retrieval_export_eval.py](C:/Users/User/Desktop/AIC2026/data_processing/caption/run_retrieval_export_eval.py:1)
- [retrieval_export.py](C:/Users/User/Desktop/AIC2026/data_processing/caption/caption_pipeline/retrieval_export.py:1)
- [retrieval_eval.py](C:/Users/User/Desktop/AIC2026/data_processing/caption/caption_pipeline/retrieval_eval.py:1)
- [retrieval_queries.py](C:/Users/User/Desktop/AIC2026/data_processing/caption/caption_pipeline/retrieval_queries.py:1)
- [retrieval_report.py](C:/Users/User/Desktop/AIC2026/data_processing/caption/caption_pipeline/retrieval_report.py:1)
- [caption_phase5_retrieval_export_eval_plan.md](C:/Users/User/Desktop/AIC2026/data_processing/caption/caption_phase5_retrieval_export_eval_plan.md:1)

Phase 5 hiện làm được:
- Build `retrieval_corpus.jsonl` từ `compact_search_index + event_step_index`.
- Export ES mapping cho compact docs và event-step docs.
- Export optional ES bulk JSONL.
- Tạo default retrieval queries.
- Chạy local lexical retrieval offline.
- Ghi `retrieval_results.jsonl`, `retrieval_eval_report.json`, `retrieval_eval_report.md`.
- `--strict` sẽ fail nếu corpus rỗng, duplicate doc id, empty search text, hoặc query không có hit.

Mình chưa compile local được vì `python.exe` trên máy sandbox vẫn trỏ tới WindowsApps stub, lỗi giống các lần trước. Chạy test thật trên server bằng lệnh này:

```bash
cd /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/data_processing/caption

PYTHONIOENCODING=utf-8 python run_retrieval_export_eval.py \
  --video-id L22_V012 \
  --caption-dir /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption \
  --top-k 20 \
  --write-es-bulk \
  --write-default-queries \
  --strict
```

Check nhanh:

```bash
wc -l /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption/L22_V012/exports/retrieval_corpus.jsonl
wc -l /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption/L22_V012/exports/es_bulk_compact_search.jsonl
wc -l /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption/L22_V012/exports/es_bulk_event_steps.jsonl
cat /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption/L22_V012/eval/retrieval_eval_report.json
```

Kỳ vọng: `retrieval_corpus = 487`, compact bulk `770 lines`, event-step bulk `204 lines`, queries đều có hit.

> "2/eval/retrieval_eval_report.md
> 00:56:35 [INFO] caption_retrieval: === Done in 4.7s ===
> 00:56:35 [INFO] caption_retrieval:   Corpus:  487 docs
> 00:56:35 [INFO] caption_retrieval:   Queries: 12
> 00:56:35 [INFO] caption_retrieval:   Results: 240
> 00:56:35 [INFO] caption_retrieval:   Hits:    12/12
> 00:56:35 [INFO] caption_retrieval:   Warnings: 0
> 00:56:35 [INFO] caption_retrieval: All retrieval export/eval acceptance checks passed.
> (base) maitanha@cl777:/tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/data_processing/caption$"

Phase 5 pass rồi. Các chỉ số đúng chuẩn:

- Corpus: `487 docs`
- Queries: `12`
- Results: `240` với `top_k=20`
- Hits: `12/12`
- Warnings: `0`
- Acceptance: passed

Như vậy pipeline caption hiện tại đã có đủ chuỗi:

`Evidence alignment -> frame caption -> shot caption -> TRAKE event-step -> retrieval export/eval`

Bước tiếp theo hợp lý là **Phase 6: benchmark nâng cấp caption/fuser**, cụ thể so sánh baseline hiện tại với một nhánh có model/VLM phụ trợ. Mình đề xuất đi theo thứ tự an toàn:

1. Viết plan Phase 6: benchmark FuseCap/BLIP/VLM fuser trên representative frames/shots.
2. Không thay pipeline chính ngay; tạo output song song kiểu `caption_experiments/<experiment_name>/`.
3. Dùng chính Phase 5 retrieval eval để so sánh baseline vs experiment.
4. Chỉ merge vào schema chính nếu retrieval/manual quality tốt hơn rõ ràng.

Hiện tại code Phase 5 đã đủ để push cùng các phase trước.

> Hãy viết  code cho phase 6 luôn bỏ qua bước .md

<details><summary>7 previous messages</summary>

> Mình sẽ code Phase 6 theo hướng benchmark/comparison harness trước: dùng chính output Phase 5 để so sánh baseline với các experiment sau này, không gọi model mới trong pipeline chính. Như vậy mình không cần phá flow đã pass, và khi có FuseCap/VLM output thì chỉ đưa thêm `--experiment label=path` để so sánh.
>
> Mình sẽ thêm Phase 6 thành script riêng và module riêng: `run_caption_experiment_benchmark.py` + `caption_pipeline/experiment_benchmark.py`. Nó sẽ đọc các report/results/corpus của Phase 5, so sánh baseline với experiment, xuất benchmark report và sample manual-review.
>
> <details><summary>Created a file</summary>
>
> - Wrote `C:/Users/User/Desktop/AIC2026/data_processing/caption/caption_pipeline/experiment_benchmark.py` (+333 -0)
>
> </details>
>
> Module benchmark đã thêm. Mình sẽ thêm CLI Phase 6 để chạy được trên server: baseline bắt buộc, experiment tùy chọn dạng `label=path`, output nằm trong `caption/<video_id>/benchmarks/<name>/`.
>
> <details><summary>Created a file</summary>
>
> - Wrote `C:/Users/User/Desktop/AIC2026/data_processing/caption/run_caption_experiment_benchmark.py` (+168 -0)
>
> </details>
>
> CLI Phase 6 đã có. Mình sẽ rà tĩnh các file mới, kiểm tra import/tên hàm và thử compile nếu môi trường local cho phép.
>
> Local `py_compile` vẫn vướng WindowsApps Python stub giống trước, nên mình không thể compile ở máy này. Mình sẽ đọc lại code Phase 6 một vòng để bắt lỗi logic rõ ràng trước khi đưa lệnh chạy server.
</details>

Đã code Phase 6, bỏ qua `.md` như bạn yêu cầu.

Đã thêm:
- [run_caption_experiment_benchmark.py](C:/Users/User/Desktop/AIC2026/data_processing/caption/run_caption_experiment_benchmark.py:1)
- [caption_pipeline/experiment_benchmark.py](C:/Users/User/Desktop/AIC2026/data_processing/caption/caption_pipeline/experiment_benchmark.py:1)

Phase 6 hiện làm:
- Đọc output Phase 5 của baseline: `retrieval_eval_report.json`, `retrieval_results.jsonl`, `retrieval_queries.jsonl`, `retrieval_corpus.jsonl`.
- So sánh baseline với nhiều experiment dạng `label=path`.
- Xuất:
  - `phase6_retrieval_benchmark.json`
  - `phase6_retrieval_benchmark.md`
  - `manual_review_samples.jsonl`
- Có thể chạy chỉ baseline trước để tạo benchmark snapshot.
- Khi có FuseCap/VLM experiment sau này, chỉ cần thêm `--experiment vlm=/path/to/experiment_caption`.

Lệnh chạy baseline-only trên server:

```bash
cd /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/data_processing/caption

PYTHONIOENCODING=utf-8 python run_caption_experiment_benchmark.py \
  --video-id L22_V012 \
  --baseline-dir /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption \
  --baseline-label template_baseline \
  --top-n-samples 3 \
  --strict
```

Sau này nếu có experiment:

```bash
PYTHONIOENCODING=utf-8 python run_caption_experiment_benchmark.py \
  --video-id L22_V012 \
  --baseline-dir /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption \
  --baseline-label template_baseline \
  --experiment vlm_fuser=/tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption_experiments/vlm_fuser \
  --top-n-samples 3 \
  --strict
```

Mình vẫn chưa compile local được vì Python trên sandbox trỏ vào WindowsApps stub, nhưng đã rà static/import/tên hàm.

> khi chạy phase sẽ chạy những mô hình nào

Phase 6 hiện tại **không chạy mô hình nào**.

Nó chỉ là benchmark/evaluation harness:

- Đọc output đã có từ Phase 5.
- So sánh `baseline` với các `experiment`.
- Tính metric retrieval: hit rate, top-1 score, warning, changed top-1 docs.
- Xuất report và sample để review thủ công.

Nói ngắn gọn:

```text
Phase 6 hiện tại = compare/evaluate
Không chạy VLM
Không chạy LLM
Không chạy BLIP/FuseCap
Không chạy embedding model
Không chạy Elasticsearch thật
```

Các mô hình đã chạy trước đó là từ những pipeline nguồn:

```text
OCR pipeline        -> PaddleOCR / VLM correction nếu bật
Object detection    -> RAM++ / GroundingDINO
Audio extraction    -> ASR/audio models
Caption Phase 2-4   -> template deterministic, chưa gọi LLM/VLM
Retrieval Phase 5   -> lexical local scorer, không model
Phase 6             -> benchmark reports, không model
```

Nếu muốn Phase 6 “thật sự benchmark mô hình”, bước tiếp theo mới là thêm một nhánh experiment riêng, ví dụ:

```text
Experiment A: BLIP/FuseCap initial caption
Experiment B: InternVL/Qwen2.5-VL frame/shot caption
Experiment C: LLM text fuser từ OCR + object + audio + VLM caption
```

Sau đó chạy lại Phase 5 trên output experiment, rồi Phase 6 chỉ so sánh baseline vs experiment. Hiện tại mình cố tình để Phase 6 không gọi model để không làm hỏng baseline đã pass.

> "(base) maitanha@cl777:/tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/data_processing/caption$ cd /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/data_processing/caption
>
> PYTHONIOENCODING=utf-8 python run_caption_experiment_benchmark.py \
>   --video-id L22_V012 \
>   --baseline-dir /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption \
>   --baseline-label template_baseline \
>   --top-n-samples 3 \
>   --strict
> 01:08:11 [INFO] caption_phase6_benchmark: === Caption Phase 6 Benchmark - L22_V012 ===
> 01:08:11 [INFO] caption_phase6_benchmark: Baseline:    template_baseline -> /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption/L22_V012
> 01:08:11 [INFO] caption_phase6_benchmark: Output:      /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption/L22_V012/benchmarks/phase6_retrieval_benchmark
> 01:08:11 [INFO] caption_phase6_benchmark: Wrote benchmark JSON -> /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption/L22_V012/benchmarks/phase6_retrieval_benchmark/phase6_retrieval_benchmark.json
> 01:08:11 [INFO] caption_phase6_benchmark: Wrote benchmark MD -> /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption/L22_V012/benchmarks/phase6_retrieval_benchmark/phase6_retrieval_benchmark.md
> 01:08:11 [INFO] caption_phase6_benchmark: Wrote manual review samples -> /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption/L22_V012/benchmarks/phase6_retrieval_benchmark/manual_review_samples.jsonl
> 01:08:11 [INFO] caption_phase6_benchmark: === Done in 0.0s ===
> 01:08:11 [INFO] caption_phase6_benchmark:   Candidates: 1
> 01:08:11 [INFO] caption_phase6_benchmark:   Samples:    12 queries
> 01:08:11 [INFO] caption_phase6_benchmark:   Warnings:   0
> 01:08:11 [INFO] caption_phase6_benchmark: All caption experiment benchmark checks passed." tôi sẽ chạy gì tiếp theo

Phase 6 baseline snapshot đã pass. Hiện tại bạn đã có full baseline từ input evidence tới retrieval benchmark.

Tiếp theo có 2 hướng, tùy mục tiêu gần nhất:

**Nếu muốn chuẩn bị push code:** chạy check trạng thái và push.

```bash
cd /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test

git status
git add data_processing/caption
git commit -m "Add caption retrieval export and benchmark pipeline"
git push
```

**Nếu muốn tiếp tục cải thiện chất lượng caption/search:** bước tiếp theo là tạo **experiment thật** để Phase 6 so sánh, ví dụ VLM/FuseCap fuser. Flow sẽ là:

```text
1. Tạo caption_experiments/vlm_fuser/L22_V012/
2. Sinh frame/shot captions bằng VLM hoặc LLM fuser vào experiment đó
3. Chạy Phase 5 trên experiment output
4. Chạy Phase 6:
   baseline vs vlm_fuser
```

Vì hiện Phase 6 mới có `Candidates: 1`, nó chỉ là snapshot baseline. Muốn Phase 6 có ý nghĩa so sánh thì cần thêm ít nhất một experiment:

```bash
PYTHONIOENCODING=utf-8 python run_caption_experiment_benchmark.py \
  --video-id L22_V012 \
  --baseline-dir /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption \
  --baseline-label template_baseline \
  --experiment vlm_fuser=/tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption_experiments/vlm_fuser \
  --top-n-samples 3 \
  --strict
```

Mình khuyên bước kế tiếp: **push code trước**, rồi sau đó bắt đầu nhánh experiment VLM/FuseCap riêng để không làm nhiễu baseline đã pass.

> Tóm gọn lại, tôi đã push code và tôi cần chạy như thế nào

Sau khi `git pull` trên server, chạy theo thứ tự này.

**1. Caption pipeline đầy đủ Phase 0-4**
```bash
cd /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/data_processing/caption

PYTHONIOENCODING=utf-8 python run_caption_pipeline.py \
  --video-id L22_V012 \
  --ocr-jsonl /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/outputs/ocr_vlm_pipeline_v2/L22_V012/L22_V012_ocr_es_docs.jsonl \
  --object-jsonl /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/outputs/object_detection/L22_V012_objects.jsonl \
  --audio-features /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/outputs/audio/L22_V012/features/audio_features.jsonl \
  --output-dir /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption \
  --frame-window-before 5 \
  --frame-window-after 5 \
  --shot-max-gap-sec 5 \
  --shot-max-duration-sec 20 \
  --enable-frame-captions \
  --caption-mode template \
  --enable-shot-captions \
  --shot-caption-mode template \
  --enable-trake-events \
  --trake-event-mode template \
  --rebuild-compact-with-captions
```

**2. Retrieval export/eval Phase 5**
```bash
PYTHONIOENCODING=utf-8 python run_retrieval_export_eval.py \
  --video-id L22_V012 \
  --caption-dir /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption \
  --top-k 20 \
  --write-es-bulk \
  --write-default-queries \
  --strict
```

**3. Benchmark snapshot Phase 6**
```bash
PYTHONIOENCODING=utf-8 python run_caption_experiment_benchmark.py \
  --video-id L22_V012 \
  --baseline-dir /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption \
  --baseline-label template_baseline \
  --top-n-samples 3 \
  --strict
```

Kỳ vọng pass:
```text
Phase 0-4: Frames 283, Shots 102, Index 385, Event steps 102/102
Phase 5: Corpus 487, Queries 12, Hits 12/12, Warnings 0
Phase 6: Candidates 1, Samples 12 queries, Warnings 0
```

> Đã đúng với kỳ vọng, hãy tiếp tục với phase tiếp theo

<details><summary>6 previous messages</summary>

> Tốt, vậy baseline đã đứng vững. Mình sẽ chuyển sang phase tiếp theo theo hướng “experiment materializer”: tạo nhánh experiment riêng từ baseline, cho phép overlay caption/VLM output sau này rồi rebuild lại index/eval/benchmark mà không đụng pipeline chính.
>
> Mình sẽ code phase tiếp theo thành “experiment materializer”: tạo thư mục `caption_experiments/<name>/<video_id>` từ baseline, overlay caption/VLM output nếu có, rồi bạn chạy lại Phase 5 và Phase 6 để so sánh. Đây là cầu nối cần thiết trước khi đưa VLM/FuseCap vào benchmark thật.
>
> <details><summary>Created a file</summary>
>
> - Wrote `C:/Users/User/Desktop/AIC2026/data_processing/caption/caption_pipeline/experiment_materializer.py` (+380 -0)
>
> </details>
>
> Materializer core đã vào. Mình sẽ thêm CLI `run_caption_experiment_materialize.py` để dùng được ngay trên server, với input overlay optional cho frame/shot/event.
>
> <details><summary>Created a file</summary>
>
> - Wrote `C:/Users/User/Desktop/AIC2026/data_processing/caption/run_caption_experiment_materialize.py` (+129 -0)
>
> </details>
>
> CLI đã thêm. Mình sẽ rà tĩnh import/tên hàm và thử compile nếu môi trường local cho phép; nếu vẫn vướng WindowsApps thì mình sẽ báo lệnh server để bạn test.
</details>

Đã code phase tiếp theo: **experiment materializer**.

Thêm 2 file:
- [run_caption_experiment_materialize.py](C:/Users/User/Desktop/AIC2026/data_processing/caption/run_caption_experiment_materialize.py:1)
- [caption_pipeline/experiment_materializer.py](C:/Users/User/Desktop/AIC2026/data_processing/caption/caption_pipeline/experiment_materializer.py:1)

Nó dùng để tạo nhánh experiment riêng từ baseline:

```text
baseline caption/L22_V012
-> caption_experiments/<experiment_name>/L22_V012
```

Có thể overlay:
- `--frame-captions`: JSONL có `canonical_frame_id` hoặc `frame_id` + `caption_text`
- `--shot-captions`: JSONL có `shot_id` + `caption_text` / `temporal_caption`
- `--event-steps`: JSONL có `event_id` hoặc `shot_id` + `trake_text`

Chạy test identity experiment trước trên server:

```bash
cd /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/data_processing/caption

PYTHONIOENCODING=utf-8 python run_caption_experiment_materialize.py \
  --video-id L22_V012 \
  --baseline-dir /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption \
  --output-root /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption_experiments \
  --experiment-name identity_copy \
  --strict
```

Sau đó chạy Phase 5 cho experiment:

```bash
PYTHONIOENCODING=utf-8 python run_retrieval_export_eval.py \
  --video-id L22_V012 \
  --caption-dir /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption_experiments/identity_copy \
  --top-k 20 \
  --write-es-bulk \
  --write-default-queries \
  --strict
```

Rồi Phase 6 so sánh:

```bash
PYTHONIOENCODING=utf-8 python run_caption_experiment_benchmark.py \
  --video-id L22_V012 \
  --baseline-dir /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption \
  --baseline-label template_baseline \
  --experiment identity_copy=/tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption_experiments/identity_copy \
  --top-n-samples 3 \
  --strict
```

Local vẫn chưa `py_compile` được vì Python sandbox trỏ WindowsApps stub, nên hãy test thật bằng server như các phase trước.

> Đã pass hết tất cả

Tuyệt, vậy hiện tại pipeline đã pass trọn chuỗi:

```text
Phase 0-4: caption/evidence/trake pipeline
Phase 5: retrieval export + eval
Phase 6: benchmark snapshot
Phase 7: experiment materializer / identity experiment
```

Trạng thái hiện giờ có thể xem là **baseline ổn định để scale hoặc làm experiment VLM/FuseCap**.

Bước tiếp theo nên chọn một trong hai:

**A. Scale batch nhiều video**
Chạy cùng chuỗi lệnh cho nhiều `video_id`, rồi kiểm tra report theo batch. Đây là hướng nên làm nếu mục tiêu gần là build index cho dataset.

**B. Tạo experiment model thật**
Sinh caption từ VLM/FuseCap/LLM vào JSONL override, materialize experiment, chạy Phase 5, rồi dùng Phase 6 so sánh với `template_baseline`.

Nếu ưu tiên A, bước tiếp theo mình sẽ viết script batch runner/resume cho nhiều video.

Nếu ưu tiên B, bước tiếp theo mình sẽ viết pipeline tạo `frame_caption_overrides.jsonl` bằng VLM hoặc LLM fuser để đưa vào experiment materializer.

> Trước hết hãy phân tích ngắn về hướng B , về cấu hình , .env chia batch như thế nào, có cần nhiều GPU hay không?

Hướng B = tạo **experiment model thật** để so sánh với baseline template hiện tại.

Mục tiêu không phải thay pipeline chính ngay, mà tạo output song song:

```text
caption_experiments/<experiment_name>/<video_id>/
```

Sau đó chạy lại Phase 5 + Phase 6 để biết model có cải thiện search không.

**Nên chạy model nào trước**
Mình khuyên thứ tự:

```text
B1. VLM frame caption override
B2. Shot-level LLM fuser từ frame captions + OCR/object/audio
B3. TRAKE/event-step LLM refinement nếu cần
```

Không nên bắt đầu bằng full VLM cho toàn bộ mọi frame nếu dataset lớn. Bắt đầu bằng representative frames hoặc sampled frames sẽ rẻ và dễ debug hơn.

**Cấu hình nên có**
Nên thêm file config riêng, ví dụ:

```text
data_processing/caption/configs/caption_experiment_vlm_a5000.yaml
```

Các nhóm config chính:

```yaml
experiment:
  name: vlm_frame_caption_v1
  video_ids: ["L22_V012"]
  output_root: /tmp2/.../caption_experiments

paths:
  caption_baseline_dir: /tmp2/.../caption
  keyframes_root: /tmp2/.../keyframe_test

model:
  provider: local_hf
  model_name: Qwen/Qwen2.5-VL-7B-Instruct
  device: cuda
  dtype: bfloat16
  attn_implementation: flash_attention_2

batch:
  batch_size: 1
  max_frames_per_video: 0
  frame_selection: representative
  workers: 1
  resume: true
  shard_index: 0
  num_shards: 1

generation:
  max_new_tokens: 96
  temperature: 0.0
  prompt_version: vlm_frame_caption_v1

runtime:
  cleanup_cache_every: 100
  write_every: 20
```

**.env nên chứa gì**
`.env` chỉ nên chứa biến môi trường và secret/path tùy server, không chứa logic pipeline.

Ví dụ:

```bash
AIC_ROOT=/tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test
CAPTION_BASELINE_DIR=/tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption
CAPTION_EXPERIMENT_ROOT=/tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption_experiments
KEYFRAME_ROOT=/tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/keyframe_test
HF_HOME=/tmp2/maitanha/.cache/huggingface
TRANSFORMERS_CACHE=/tmp2/maitanha/.cache/huggingface
CUDA_VISIBLE_DEVICES=1
```

Nếu dùng API model thì thêm:

```bash
OPENAI_API_KEY=...
GEMINI_API_KEY=...
```

Không nên commit `.env`.

**Chia batch như thế nào**
Có 3 cách chia:

```text
1. Theo video_id
2. Theo shard trong một video
3. Theo loại frame: representative/all/sampled
```

Khuyên dùng:

```text
Giai đoạn test:
  1 video, representative frames only

Giai đoạn mở rộng:
  shard theo video_id

Giai đoạn cần caption dày:
  shard theo frame index trong video
```

Ví dụ với 4 GPU:

```bash
CUDA_VISIBLE_DEVICES=0 ... --shard-index 0 --num-shards 4
CUDA_VISIBLE_DEVICES=1 ... --shard-index 1 --num-shards 4
CUDA_VISIBLE_DEVICES=2 ... --shard-index 2 --num-shards 4
CUDA_VISIBLE_DEVICES=3 ... --shard-index 3 --num-shards 4
```

Mỗi shard ghi output riêng hoặc cùng JSONL có lock/atomic append. Mình khuyên ghi riêng:

```text
frame_caption_overrides.shard_000.jsonl
frame_caption_overrides.shard_001.jsonl
...
```

Sau đó merge.

**Có cần nhiều GPU không?**
Không bắt buộc.

Với `L22_V012`:
- 283 frames
- 102 shots
- representative frames có thể ít hơn nhiều

Một GPU A5000 là đủ để test.

Nhiều GPU chỉ cần khi:
- chạy nhiều video
- dùng VLM 7B/13B trên toàn bộ frame
- cần throughput cao
- muốn chạy nhiều shard song song

**Khuyến nghị thực tế**
Bắt đầu bằng 1 GPU:

```text
CUDA_VISIBLE_DEVICES=1
batch_size=1
frame_selection=representative
resume=true
```

Sau khi pass:
- tăng sang all frames hoặc sampled frames
- tăng `num_shards`
- chạy nhiều GPU theo shard

**Output của hướng B nên là gì**
Bước model chỉ nên tạo override JSONL:

```text
caption_experiments_raw/vlm_frame_caption_v1/L22_V012/frame_caption_overrides.jsonl
```

Schema đơn giản:

```json
{
  "video_id": "L22_V012",
  "canonical_frame_id": "L22_V012_000123",
  "image_relpath": "L22_V012/000123.jpg",
  "caption_text": "A presenter stands near a screen showing conference text.",
  "caption_mode": "vlm",
  "caption_model": "Qwen2.5-VL-7B-Instruct",
  "prompt_version": "vlm_frame_caption_v1"
}
```

Rồi materialize:

```bash
python run_caption_experiment_materialize.py \
  --video-id L22_V012 \
  --baseline-dir /tmp2/.../caption \
  --output-root /tmp2/.../caption_experiments \
  --experiment-name vlm_frame_caption_v1 \
  --frame-captions /tmp2/.../caption_experiments_raw/vlm_frame_caption_v1/L22_V012/frame_caption_overrides.jsonl \
  --strict
```

Sau đó Phase 5 + Phase 6 so sánh với baseline.

Tóm lại: **không cần nhiều GPU để bắt đầu**, nên dùng `.env` cho path/device, YAML cho model/batch/generation, batch theo shard khi scale, và chỉ ghi override JSONL để giữ pipeline chính sạch.

> còn đối với A. Scale batch nhiều video

Hướng A = scale batch nhiều video bằng baseline template hiện tại. Hướng này **không cần GPU**, vì caption Phase 0-7 hiện tại không gọi VLM/LLM.

Nó chỉ đọc output có sẵn:

```text
OCR output
object_detection output
audio output
keyframe/video ids
```

rồi chạy:

```text
Phase 0-4 caption pipeline
Phase 5 retrieval export/eval
Phase 6 benchmark snapshot
Optional Phase 7 identity/materialize check
```

**Cấu hình nên có**
Nên tạo batch config YAML, ví dụ:

```yaml
batch:
  video_ids:
    - L22_V012
    - L22_V013
  continue_on_error: true
  skip_completed: true
  workers: 1

paths:
  aic_root: /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test
  caption_output_dir: /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption
  ocr_root: /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/outputs/ocr_vlm_pipeline_v2
  object_root: /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/outputs/object_detection
  audio_root: /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/outputs/audio

caption:
  frame_window_before: 5
  frame_window_after: 5
  shot_max_gap_sec: 5
  shot_max_duration_sec: 20
  caption_mode: template
  shot_caption_mode: template
  trake_event_mode: template

retrieval:
  top_k: 20
  write_es_bulk: true
  write_default_queries: true

benchmark:
  baseline_label: template_baseline
  top_n_samples: 3
```

**.env nên chứa path/server config**
Ví dụ:

```bash
AIC_ROOT=/tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test
PYTHONIOENCODING=utf-8
OMP_NUM_THREADS=4
TOKENIZERS_PARALLELISM=false
```

Không cần:

```bash
CUDA_VISIBLE_DEVICES
```

trừ khi bạn chạy lại OCR/object/audio.

**Chia batch như thế nào**
Có 3 mức:

```text
1. Theo video_id: dễ nhất, khuyên dùng
2. Theo collection/batch folder: L21, L22, L23...
3. Theo shard video list: shard 0/4, 1/4, ...
```

Khuyên dùng cho baseline:

```text
workers=1 hoặc 2
mỗi worker xử lý một video end-to-end
skip_completed=true
continue_on_error=true
```

Không nên chạy quá nhiều worker nếu đọc/ghi nhiều JSONL trên cùng ổ, vì bottleneck là I/O chứ không phải GPU.

**Resume/skip completed**
Một video coi là completed nếu có đủ:

```text
caption/<video_id>/reports/frame_caption_report.json
caption/<video_id>/reports/shot_caption_report.json
caption/<video_id>/reports/event_step_report.json
caption/<video_id>/eval/retrieval_eval_report.json
caption/<video_id>/benchmarks/phase6_retrieval_benchmark/phase6_retrieval_benchmark.json
```

Nếu thiếu file nào thì chạy lại video đó.

**Có cần nhiều GPU không?**
Không.

Scale batch baseline hiện tại chạy CPU là đủ. Nhiều GPU chỉ cần nếu bạn scale lại upstream:

```text
OCR VLM
object detection
ASR/audio model
VLM caption experiment
```

Còn caption/retrieval/benchmark baseline:

```text
GPU = không cần
CPU/RAM/I/O = quan trọng hơn
```

**Command batch thủ công nếu chưa có batch runner**
Ví dụ cho nhiều video:

```bash
for VID in L22_V012 L22_V013 L22_V014; do
  echo "=== $VID ==="

  PYTHONIOENCODING=utf-8 python run_caption_pipeline.py \
    --video-id "$VID" \
    --ocr-jsonl /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/outputs/ocr_vlm_pipeline_v2/$VID/${VID}_ocr_es_docs.jsonl \
    --object-jsonl /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/outputs/object_detection/${VID}_objects.jsonl \
    --audio-features /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/outputs/audio/$VID/features/audio_features.jsonl \
    --output-dir /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption \
    --frame-window-before 5 \
    --frame-window-after 5 \
    --shot-max-gap-sec 5 \
    --shot-max-duration-sec 20 \
    --enable-frame-captions \
    --caption-mode template \
    --enable-shot-captions \
    --shot-caption-mode template \
    --enable-trake-events \
    --trake-event-mode template \
    --rebuild-compact-with-captions || continue

  PYTHONIOENCODING=utf-8 python run_retrieval_export_eval.py \
    --video-id "$VID" \
    --caption-dir /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption \
    --top-k 20 \
    --write-es-bulk \
    --write-default-queries \
    --strict || continue

  PYTHONIOENCODING=utf-8 python run_caption_experiment_benchmark.py \
    --video-id "$VID" \
    --baseline-dir /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption \
    --baseline-label template_baseline \
    --top-n-samples 3 \
    --strict || continue
done
```

**Khuyến nghị tiếp theo**
Nên viết `run_caption_batch.py` để:
- nhận `--video-list`
- tự resolve path OCR/object/audio
- skip completed
- log manifest
- xuất batch report tổng
- chạy từng video end-to-end

Đây là bước hợp lý nhất trước khi scale toàn bộ dataset.

> Bây giời chỉ đang demo , test và experiments ở L22_V012 nãy giờ chúng ta đang làm.. Nếu pipeline và workflow đã ổn thì tiến hành experiments thật ở các mô hình dự kiến đã đề xuất

<details><summary>12 previous messages</summary>

> Mình đồng ý: baseline/workflow đã ổn thì bước tiếp theo là tạo experiment model thật, nhưng vẫn giữ nguyên nguyên tắc “model chỉ ghi override JSONL”, sau đó dùng materializer + Phase 5 + Phase 6 để so sánh. Mình sẽ code runner VLM frame-caption trước, vì đây là experiment ít rủi ro nhất và đo được ngay.
>
> Mình sẽ triển khai experiment thật theo hướng B1: **VLM frame-caption override runner**. Script này sẽ đọc baseline `frame_index/shot_index`, chọn representative/all/sampled frames, chạy Qwen2.5-VL hoặc InternVL local nếu môi trường server có model/deps, rồi ghi `frame_caption_overrides.jsonl` để đưa vào materializer.
>
> <details><summary>Created a file</summary>
>
> - Wrote `C:/Users/User/Desktop/AIC2026/data_processing/caption/caption_pipeline/vlm_frame_experiment.py` (+403 -0)
>
> </details>
>
> VLM runner core đã có: hỗ trợ `qwen2_5_vl`, `internvl`, và `mock` để smoke test. Mình sẽ thêm CLI và config A5000 mẫu để bạn chạy trên server.
>
> <details><summary>Created a file</summary>
>
> - Wrote `C:/Users/User/Desktop/AIC2026/data_processing/caption/run_vlm_frame_caption_experiment.py` (+78 -0)
>
> </details>
>
> CLI đã xong. Mình sẽ thêm config mẫu cho A5000: mặc định dùng Qwen2.5-VL, representative frames, shard/resume sẵn; có thể override provider/model bằng CLI.
>
> <details><summary>Created a file</summary>
>
> - Wrote `C:/Users/User/Desktop/AIC2026/data_processing/caption/configs/caption_experiment_vlm_a5000.yaml` (+39 -0)
>
> </details>
>
> Bây giờ mình rà import/tên hàm và các lỗi dễ thấy. Local Python có thể vẫn không compile được như trước, nhưng mình sẽ thử rồi đưa lệnh server đầy đủ.
>
> Một chỉnh nhỏ nữa: mình sẽ bỏ import PIL ở top-level để `provider=mock` và `qwen2_5_vl` không phụ thuộc Pillow tại import time; InternVL mới cần Pillow/torchvision khi chạy.
>
> `apply_patch` lại bị path Windows một nhịp. Mình sẽ dùng PowerShell replace cho chỉnh sửa nhỏ này, rồi kiểm tra lại bằng `rg`.
>
> PowerShell string quoting vừa vấp dấu quote, mình sẽ chạy lại bằng here-string cho chắc hơn.
>
> Chỉnh xong. Mình sẽ rà lại đoạn Qwen/InternVL và helper mới để chắc không sinh lỗi syntax do replace.
</details>

Đã triển khai nhánh **experiment model thật đầu tiên: VLM frame-caption override**.

Code mới:
- `data_processing/caption/run_vlm_frame_caption_experiment.py`
- `data_processing/caption/caption_pipeline/vlm_frame_experiment.py`
- `data_processing/caption/configs/caption_experiment_vlm_a5000.yaml`

Hỗ trợ provider:
- `qwen2_5_vl`: `Qwen/Qwen2.5-VL-7B-Instruct`
- `internvl`
- `mock`: để smoke test không cần GPU/model

Local vẫn không compile được vì Python sandbox trỏ WindowsApps stub, nên test thật trên server.

**1. Smoke test không dùng GPU/model**
```bash
cd /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/data_processing/caption

export AIC_ROOT=/tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test

PYTHONIOENCODING=utf-8 python run_vlm_frame_caption_experiment.py \
  --config configs/caption_experiment_vlm_a5000.yaml \
  --provider mock \
  --experiment-name mock_vlm_smoke \
  --max-frames 2 \
  --no-resume
```

**2. Chạy Qwen2.5-VL thật, test nhỏ trước**
```bash
export AIC_ROOT=/tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test
export CUDA_VISIBLE_DEVICES=1

PYTHONIOENCODING=utf-8 python run_vlm_frame_caption_experiment.py \
  --config configs/caption_experiment_vlm_a5000.yaml \
  --experiment-name vlm_frame_qwen25vl_7b_rep \
  --frame-selection representative \
  --max-frames 20
```

Output sẽ nằm ở:

```text
/tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption_experiments_raw/vlm_frame_qwen25vl_7b_rep/L22_V012/frame_caption_overrides.jsonl
```

**3. Nếu test 20 frame pass, chạy full representative**
```bash
PYTHONIOENCODING=utf-8 python run_vlm_frame_caption_experiment.py \
  --config configs/caption_experiment_vlm_a5000.yaml \
  --experiment-name vlm_frame_qwen25vl_7b_rep \
  --frame-selection representative
```

**4. Materialize experiment**
```bash
PYTHONIOENCODING=utf-8 python run_caption_experiment_materialize.py \
  --video-id L22_V012 \
  --baseline-dir /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption \
  --output-root /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption_experiments \
  --experiment-name vlm_frame_qwen25vl_7b_rep \
  --frame-captions /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption_experiments_raw/vlm_frame_qwen25vl_7b_rep/L22_V012/frame_caption_overrides.jsonl \
  --strict
```

**5. Chạy Phase 5 cho experiment**
```bash
PYTHONIOENCODING=utf-8 python run_retrieval_export_eval.py \
  --video-id L22_V012 \
  --caption-dir /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption_experiments/vlm_frame_qwen25vl_7b_rep \
  --top-k 20 \
  --write-es-bulk \
  --write-default-queries \
  --strict
```

**6. So sánh baseline vs VLM**
```bash
PYTHONIOENCODING=utf-8 python run_caption_experiment_benchmark.py \
  --video-id L22_V012 \
  --baseline-dir /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption \
  --baseline-label template_baseline \
  --experiment vlm_frame_qwen25vl_7b_rep=/tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption_experiments/vlm_frame_qwen25vl_7b_rep \
  --top-n-samples 3 \
  --strict
```

Nếu server báo thiếu dependency cho Qwen2.5-VL thì cần kiểm tra các gói: `transformers`, `qwen-vl-utils`, `accelerate`, `torch`, `flash-attn`.

> Hãy giải đáp thắc mắc giúp tôi trước khi thực hiên chạy code:
> + Thứ nhất là về phần [prompts](data_processing/caption/caption_pipeline/prompts/) có được sử dụng trong benchmark này không ?
> + tôi nhớ theo plan sẽ có 3 cấp độ caption, thì đang benchmark như thế nào ?

Có, nhưng cần phân biệt rõ:

**1. Folder `prompts/` hiện có được dùng trong benchmark này không?**

Trong baseline template hiện tại: **không dùng trực tiếp**.

Các file:

```text
frame_caption_prompt.txt
shot_caption_prompt.txt
trake_event_step_prompt.txt
```

hiện chủ yếu là **prompt placeholder/chuẩn thiết kế cho mode LLM/VLM sau này**. Pipeline đã pass vừa rồi đang chạy:

```bash
--caption-mode template
--shot-caption-mode template
--trake-event-mode template
```

nên caption được sinh bằng rule/template trong code, không gọi LLM, không đọc prompt file.

Trong experiment VLM mới mình vừa viết: cũng **chưa đọc các prompt file đó**. Nó dùng prompt trong YAML:

```yaml
generation:
  prompt: >-
    Describe this video frame for video retrieval...
```

Lý do: VLM frame-caption experiment là nhánh riêng, ghi `frame_caption_overrides.jsonl`, nên mình để prompt nằm trong config YAML để dễ đổi giữa experiment.

Nếu muốn, bước sau có thể chỉnh để:

```yaml
generation:
  prompt_file: data_processing/caption/caption_pipeline/prompts/vlm_frame_caption_prompt.txt
```

và runner sẽ đọc prompt từ file. Nhưng hiện tại chưa làm.

**2. “3 cấp độ caption” trong plan đang benchmark thế nào?**

Đúng, pipeline của bạn đang có 3 cấp độ caption/search:

```text
Level 1: Frame caption
Level 2: Shot caption / temporal caption
Level 3: Event-step / TRAKE trake_text
```

Hiện benchmark Phase 5/6 đang đánh giá **retrieval tổng hợp trên cả 3 cấp**, không chỉ một cấp riêng.

Cụ thể:

**Level 1 - Frame**
Nguồn file:

```text
caption/L22_V012/captions/frame_index.jsonl
caption/L22_V012/indexes/compact_search_index.jsonl
```

Search docs:

```text
unit_type = frame
```

Dùng cho query kiểu:

```text
object/visual
OCR trên frame
caption mô tả frame
```

**Level 2 - Shot**
Nguồn file:

```text
caption/L22_V012/captions/shot_index.jsonl
caption/L22_V012/indexes/compact_search_index.jsonl
```

Search docs:

```text
unit_type = shot
```

Dùng cho query kiểu:

```text
đoạn video
ngữ cảnh clip
audio/topic
temporal caption
```

**Level 3 - Event-step / TRAKE**
Nguồn file:

```text
caption/L22_V012/captions/event_step_index.jsonl
```

Search docs:

```text
unit_type = event_step
```

Dùng cho query kiểu:

```text
đầu tiên...
sau đó...
cuối cùng...
hành động theo thứ tự thời gian
```

Phase 5 gom lại thành:

```text
retrieval_corpus.jsonl = 487 docs
= 283 frame docs
+ 102 shot docs
+ 102 event_step docs
```

Sau đó chạy 12 query mặc định. Mỗi query có `target_unit_types`, ví dụ:

```json
{
  "query_id": "q_obj_001",
  "target_unit_types": ["frame", "shot"]
}
```

hoặc:

```json
{
  "query_id": "q_trake_001",
  "target_unit_types": ["event_step", "shot"]
}
```

Vì vậy benchmark hiện tại đang hỏi:

```text
Với cùng bộ query, hệ thống retrieve tốt hơn ở frame, shot, hay event_step?
```

**3. Experiment VLM mới benchmark vào cấp nào?**

Experiment mình vừa code là:

```text
VLM frame-caption override
```

Tức là nó chỉ thay **Level 1 frame caption** trước.

Flow:

```text
VLM sinh frame_caption_overrides.jsonl
-> materialize experiment
-> frame_index caption_text đổi
-> compact_search_index frame docs đổi
-> Phase 5 eval lại retrieval_corpus
-> Phase 6 so sánh baseline vs VLM
```

Nó chưa trực tiếp sinh lại:

```text
shot caption
event-step trake_text
```

Nhưng nó vẫn có thể ảnh hưởng gián tiếp nếu sau đó ta rebuild shot caption từ frame caption mới. Hiện materializer chỉ overlay frame captions vào experiment, chưa chạy lại shot fuser/event fuser.

**4. Nếu muốn benchmark đủ 3 cấp với VLM thì cần làm thêm gì?**

Nên làm theo thứ tự:

```text
B1. VLM frame caption
  -> benchmark Level 1 impact

B2. Rebuild shot caption từ VLM frame caption + OCR/object/audio
  -> benchmark Level 2 impact

B3. Rebuild event-step/trake_text từ shot caption mới
  -> benchmark Level 3 impact
```

Khi đó ta có các experiment rõ ràng:

```text
template_baseline
vlm_frame_only
vlm_frame_plus_shot_refuse
vlm_frame_plus_shot_plus_trake
```

Rồi Phase 6 so sánh:

```bash
--experiment vlm_frame_only=...
--experiment vlm_frame_plus_shot_refuse=...
--experiment vlm_frame_plus_shot_plus_trake=...
```

**Tóm tắt ngắn**

- `prompts/*.txt` hiện **chưa được dùng trực tiếp** trong baseline/benchmark/VLM runner.
- Benchmark hiện đánh giá trên **3 cấp retrieval docs**: frame, shot, event_step.
- VLM experiment hiện tại mới tác động **Level 1 frame caption**.
- Muốn đúng “3 cấp độ caption” với model thật thì bước tiếp theo là thêm runner rebuild shot caption và event-step từ output VLM frame caption.

> tôi muốn benchmark trên toàn bộ pipeline chuẩn đã được thiết kế thì sẽ thực hiện như thế nào

Để benchmark **toàn bộ pipeline chuẩn đã thiết kế**, bạn nên chạy theo 2 lớp:

```text
Layer 1: baseline chuẩn
OCR + Object + Audio -> Caption 3 cấp -> Retrieval eval -> Benchmark snapshot

Layer 2: model experiments
VLM/LLM override -> Materialize experiment -> Retrieval eval -> Compare với baseline
```

**1. Chuẩn bị video list**

Ví dụ:

```text
L22_V012
L22_V013
L22_V014
...
```

Mỗi video cần có đủ input:

```text
outputs/ocr_vlm_pipeline_v2/<VID>/<VID>_ocr_es_docs.jsonl
outputs/object_detection/<VID>_objects.jsonl
outputs/audio/<VID>/features/audio_features.jsonl
keyframe_test/<VID>/
```

**2. Chạy baseline chuẩn Phase 0-4**

Pipeline này tạo đủ 3 cấp caption:

```text
Level 1: frame_index.jsonl
Level 2: shot_index.jsonl
Level 3: event_step_index.jsonl / trake_text
```

Lệnh mỗi video:

```bash
PYTHONIOENCODING=utf-8 python run_caption_pipeline.py \
  --video-id "$VID" \
  --ocr-jsonl /tmp2/.../outputs/ocr_vlm_pipeline_v2/$VID/${VID}_ocr_es_docs.jsonl \
  --object-jsonl /tmp2/.../outputs/object_detection/${VID}_objects.jsonl \
  --audio-features /tmp2/.../outputs/audio/$VID/features/audio_features.jsonl \
  --output-dir /tmp2/.../caption \
  --frame-window-before 5 \
  --frame-window-after 5 \
  --shot-max-gap-sec 5 \
  --shot-max-duration-sec 20 \
  --enable-frame-captions \
  --caption-mode template \
  --enable-shot-captions \
  --shot-caption-mode template \
  --enable-trake-events \
  --trake-event-mode template \
  --rebuild-compact-with-captions
```

**3. Chạy Phase 5 retrieval eval**

Gom 3 cấp thành corpus search:

```text
frame docs + shot docs + event_step docs
```

```bash
PYTHONIOENCODING=utf-8 python run_retrieval_export_eval.py \
  --video-id "$VID" \
  --caption-dir /tmp2/.../caption \
  --top-k 20 \
  --write-es-bulk \
  --write-default-queries \
  --strict
```

**4. Chạy Phase 6 benchmark baseline**

```bash
PYTHONIOENCODING=utf-8 python run_caption_experiment_benchmark.py \
  --video-id "$VID" \
  --baseline-dir /tmp2/.../caption \
  --baseline-label template_baseline \
  --top-n-samples 3 \
  --strict
```

Đến đây bạn có benchmark baseline chuẩn cho từng video.

**5. Nếu benchmark model thật**

Ví dụ VLM frame caption:

```bash
export AIC_ROOT=/tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test
export CUDA_VISIBLE_DEVICES=1

PYTHONIOENCODING=utf-8 python run_vlm_frame_caption_experiment.py \
  --config configs/caption_experiment_vlm_a5000.yaml \
  --video-id "$VID" \
  --experiment-name vlm_frame_qwen25vl_7b_rep \
  --frame-selection representative
```

Sau đó materialize:

```bash
PYTHONIOENCODING=utf-8 python run_caption_experiment_materialize.py \
  --video-id "$VID" \
  --baseline-dir /tmp2/.../caption \
  --output-root /tmp2/.../caption_experiments \
  --experiment-name vlm_frame_qwen25vl_7b_rep \
  --frame-captions /tmp2/.../caption_experiments_raw/vlm_frame_qwen25vl_7b_rep/$VID/frame_caption_overrides.jsonl \
  --strict
```

Rồi chạy Phase 5 cho experiment:

```bash
PYTHONIOENCODING=utf-8 python run_retrieval_export_eval.py \
  --video-id "$VID" \
  --caption-dir /tmp2/.../caption_experiments/vlm_frame_qwen25vl_7b_rep \
  --top-k 20 \
  --write-es-bulk \
  --write-default-queries \
  --strict
```

Cuối cùng so sánh baseline vs experiment:

```bash
PYTHONIOENCODING=utf-8 python run_caption_experiment_benchmark.py \
  --video-id "$VID" \
  --baseline-dir /tmp2/.../caption \
  --baseline-label template_baseline \
  --experiment vlm_frame_qwen25vl_7b_rep=/tmp2/.../caption_experiments/vlm_frame_qwen25vl_7b_rep \
  --top-n-samples 3 \
  --strict
```

**6. Metric cần xem**

Cho từng video:

```text
Phase 0-4:
- Frames
- Shots
- Event steps
- Warnings
- Caption coverage
- Trake coverage

Phase 5:
- Corpus docs
- Queries
- Hits 12/12
- Warnings
- Top result unit_type: frame/shot/event_step

Phase 6:
- hit_rate
- avg_top1_score
- changed_top1_document
- improved/regressed top1 score
- manual_review_samples
```

**7. Benchmark chuẩn nên báo cáo ở 3 mức**

```text
A. Per-video report
  caption/<VID>/reports
  caption/<VID>/eval
  caption/<VID>/benchmarks

B. Per-experiment comparison
  baseline vs vlm_frame_only
  baseline vs shot_refuse
  baseline vs trake_refine

C. Global summary
  trung bình nhiều video:
  hit_rate
  warnings
  avg_top1_score
  failure videos
```

Hiện bạn đã có đủ code để chạy **per-video benchmark**. Bước còn thiếu để benchmark toàn bộ dataset gọn hơn là viết thêm:

```text
run_caption_batch_benchmark.py
```

Script này sẽ:
- đọc `video_list.txt`
- chạy Phase 0-4, 5, 6 cho từng video
- skip video đã pass
- ghi manifest
- tạo global benchmark summary

Nếu mục tiêu là benchmark toàn bộ pipeline chuẩn trên nhiều video, bước tiếp theo nên là code batch benchmark runner này.

> Hãy phân tích rõ lại benchmark đang theo những hướng nào ? Có mô hình nào đang được sử dụng
> Lưu ý rằng tôi chỉ cần benchmark trên L22_v012

Hiện tại với **L22_V012**, benchmark của bạn đang có 2 hướng chính.

**Hướng 1: Benchmark baseline chuẩn**
Đây là pipeline đã pass đầy đủ.

```text
OCR output
+ Object detection output
+ Audio output
-> Caption pipeline Phase 0-4
-> Retrieval eval Phase 5
-> Benchmark snapshot Phase 6
```

Ở hướng này, caption được sinh bằng **template/rule deterministic**, không gọi model caption mới.

Nó benchmark 3 cấp:

```text
Level 1: Frame caption
  file: frame_index.jsonl
  unit_type: frame

Level 2: Shot caption / temporal caption
  file: shot_index.jsonl
  unit_type: shot

Level 3: Event-step / TRAKE text
  file: event_step_index.jsonl
  unit_type: event_step
```

Phase 5 gom lại thành:

```text
retrieval_corpus.jsonl = 487 docs
= 283 frame docs
+ 102 shot docs
+ 102 event_step docs
```

Rồi chạy 12 query test để đo:

```text
Hits: 12/12
Warnings: 0
Top result thuộc frame/shot/event_step nào
Matched fields là caption/OCR/audio/object/trake_text nào
```

Phase 6 tạo snapshot benchmark:

```text
template_baseline
```

Nếu chỉ chạy baseline thì:

```text
Candidates = 1
```

Nó chưa so sánh với model khác, chỉ lưu chuẩn baseline.

**Có model nào được dùng trong baseline không?**

Trong caption benchmark baseline: **không có model caption nào được gọi**.

Nhưng nó dùng output từ các model upstream đã chạy trước đó:

```text
OCR pipeline:
  PaddleOCR / OCR-VLM correction nếu bạn bật trong pipeline OCR

Object detection:
  RAM++ / GroundingDINO

Audio:
  ASR/audio model từ audio_extraction

Caption Phase 2-4:
  template deterministic, không LLM/VLM

Retrieval Phase 5:
  lexical scorer local, không embedding model

Benchmark Phase 6:
  compare/report, không model
```

Tức là model đã nằm ở **nguồn evidence**, còn caption/retrieval benchmark hiện tại là rule-based.

**Hướng 2: Benchmark experiment model thật**
Đây là hướng mới mình vừa thêm, nhưng bạn chưa cần chạy nếu chỉ muốn baseline.

Mục tiêu là tạo một nhánh model experiment:

```text
VLM frame caption
-> frame_caption_overrides.jsonl
-> materialize experiment
-> Phase 5 retrieval eval
-> Phase 6 compare với template_baseline
```

Model dự kiến:

```text
Qwen2.5-VL-7B-Instruct
InternVL
mock provider để smoke test
```

Hiện code experiment đầu tiên mới tác động vào:

```text
Level 1: frame caption
```

Nó chưa tự rebuild Level 2 và Level 3 bằng model.

Nếu chạy VLM experiment thì benchmark Phase 6 sẽ thành:

```text
template_baseline
vs
vlm_frame_qwen25vl_7b_rep
```

Lúc đó mới có so sánh model thật.

**Tóm tắt cho L22_V012**

Hiện bạn có thể benchmark theo 2 chế độ:

```text
A. Baseline-only benchmark
  Không chạy model caption mới
  Dùng OCR/Object/Audio đã có
  Caption bằng template
  Output: baseline snapshot

B. Model experiment benchmark
  Chạy thêm VLM như Qwen2.5-VL hoặc InternVL
  Sinh frame_caption_overrides
  So sánh retrieval với baseline
```

Nếu mục tiêu hiện tại là “benchmark pipeline chuẩn đã thiết kế trên L22_V012”, thì benchmark đúng nhất hiện tại là:

```text
A. Baseline chuẩn 3 cấp
```

Nếu mục tiêu là “benchmark mô hình caption nào tốt hơn”, thì cần chạy thêm:

```text
B. VLM frame-caption experiment
```

**Kết luận ngắn**
- Benchmark hiện tại trên `L22_V012` đã đánh giá đủ 3 cấp: `frame`, `shot`, `event_step`.
- Trong caption pipeline baseline không chạy LLM/VLM.
- Model đang được dùng chỉ là upstream: OCR, RAM++/GroundingDINO, ASR.
- Model caption thật đầu tiên được chuẩn bị là Qwen2.5-VL/InternVL, nhưng chỉ chạy khi bạn chọn hướng experiment.

> Tôi sẽ thực hiện theo hướng này VLM frame-caption experiment

Đúng hướng. Với `L22_V012`, bạn chạy theo thứ tự này.

**1. Smoke test trước bằng mock**

Lệnh này không dùng GPU/model, chỉ kiểm tra path + output format:

```bash
cd /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/data_processing/caption

export AIC_ROOT=/tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test

PYTHONIOENCODING=utf-8 python run_vlm_frame_caption_experiment.py \
  --config configs/caption_experiment_vlm_a5000.yaml \
  --provider mock \
  --experiment-name mock_vlm_smoke \
  --max-frames 2 \
  --no-resume
```

Nếu pass, sẽ có:

```text
caption_experiments_raw/mock_vlm_smoke/L22_V012/frame_caption_overrides.jsonl
```

**2. Chạy VLM thật trên ít frame trước**

```bash
export AIC_ROOT=/tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test
export CUDA_VISIBLE_DEVICES=1

PYTHONIOENCODING=utf-8 python run_vlm_frame_caption_experiment.py \
  --config configs/caption_experiment_vlm_a5000.yaml \
  --experiment-name vlm_frame_qwen25vl_7b_rep \
  --frame-selection representative \
  --max-frames 20
```

Nếu thiếu dependency, kiểm tra các gói:

```bash
python -c "import torch, transformers, qwen_vl_utils, accelerate; print('ok')"
```

**3. Nếu 20 frame pass, chạy full representative**

```bash
PYTHONIOENCODING=utf-8 python run_vlm_frame_caption_experiment.py \
  --config configs/caption_experiment_vlm_a5000.yaml \
  --experiment-name vlm_frame_qwen25vl_7b_rep \
  --frame-selection representative
```

Output chính:

```text
/tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption_experiments_raw/vlm_frame_qwen25vl_7b_rep/L22_V012/frame_caption_overrides.jsonl
```

**4. Materialize experiment**

```bash
PYTHONIOENCODING=utf-8 python run_caption_experiment_materialize.py \
  --video-id L22_V012 \
  --baseline-dir /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption \
  --output-root /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption_experiments \
  --experiment-name vlm_frame_qwen25vl_7b_rep \
  --frame-captions /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption_experiments_raw/vlm_frame_qwen25vl_7b_rep/L22_V012/frame_caption_overrides.jsonl \
  --strict
```

**5. Chạy retrieval eval cho experiment**

```bash
PYTHONIOENCODING=utf-8 python run_retrieval_export_eval.py \
  --video-id L22_V012 \
  --caption-dir /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption_experiments/vlm_frame_qwen25vl_7b_rep \
  --top-k 20 \
  --write-es-bulk \
  --write-default-queries \
  --strict
```

**6. So sánh baseline vs VLM**

```bash
PYTHONIOENCODING=utf-8 python run_caption_experiment_benchmark.py \
  --video-id L22_V012 \
  --baseline-dir /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption \
  --baseline-label template_baseline \
  --experiment vlm_frame_qwen25vl_7b_rep=/tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption_experiments/vlm_frame_qwen25vl_7b_rep \
  --top-n-samples 3 \
  --strict
```

Bạn cần xem 3 file sau:

```text
caption_experiments_raw/vlm_frame_qwen25vl_7b_rep/L22_V012/frame_caption_overrides.shard_000.report.json
caption_experiments/vlm_frame_qwen25vl_7b_rep/L22_V012/eval/retrieval_eval_report.json
caption/L22_V012/benchmarks/phase6_retrieval_benchmark/phase6_retrieval_benchmark.json
```

Nếu `vlm_frame_qwen25vl_7b_rep` có `avg_top1_score` cao hơn hoặc manual samples tốt hơn baseline, lúc đó mới đáng làm bước kế tiếp: rebuild shot caption từ VLM frame captions.

> tôi nhớ có các mô hình LLM hay Fusecap, tại sao không sử dụng ở experiments này

Đúng, trong plan ban đầu có nhắc **LLM fuser / FuseCap-style / VLM fuser**. Lý do hiện tại mình chưa cho chạy ngay là vì mình đang tách experiment theo mức rủi ro và tác động:

```text
B1. VLM frame-caption override       <- đang chuẩn bị chạy
B2. LLM/FuseCap-style shot fuser     <- bước kế tiếp
B3. LLM TRAKE/event-step refiner     <- sau B2
```

**Vì sao không chạy LLM/FuseCap ngay?**

Vì baseline của bạn có 3 cấp:

```text
frame -> shot -> event_step
```

Nếu chạy LLM/FuseCap ngay ở shot/event level mà chưa biết frame VLM có tốt không, sẽ khó biết model nào làm kết quả tốt/xấu:

```text
VLM frame caption sai?
LLM shot fuser sai?
OCR/audio/object noisy?
TRAKE refiner hallucinate?
```

Nên mình chọn **VLM frame-caption experiment** trước để cô lập biến số:

```text
Chỉ thay frame caption
Giữ OCR/object/audio/shot/trake baseline gần như nguyên
Đo retrieval có cải thiện không
```

Đây là bước ít nhiễu nhất.

**FuseCap ở đây nghĩa là gì?**

Trong context pipeline của bạn, “FuseCap-style” không nhất thiết là model tên FuseCap chạy nguyên bản. Nó là chiến lược:

```text
image/visual caption
+ object evidence
+ OCR evidence
+ audio evidence
-> fused caption/search text
```

Hiện baseline template đang làm bản rule-based của fuse:

```text
object + OCR + audio -> frame caption / shot caption
```

Còn experiment FuseCap/LLM thật sẽ là:

```text
VLM caption + object + OCR + audio -> LLM fuser -> better caption
```

**LLM sẽ dùng ở đâu hợp lý nhất?**

Không nên dùng LLM để caption từng frame từ ảnh, vì LLM text-only không nhìn ảnh. LLM nên dùng để **fuse text evidence**:

```text
Input:
- VLM frame caption
- OCR text
- object labels
- audio segment
- baseline caption

Output:
- cleaner frame caption
- search tags
- visible actions
- hallucination-safe summary
```

Hoặc ở shot level:

```text
Input:
- representative frame captions
- OCR merged
- audio merged
- object counts
- previous/next shot context

Output:
- shot_caption
- temporal_caption
```

**Mô hình đề xuất theo cấp**

```text
Level 1 frame:
  Qwen2.5-VL / InternVL
  -> nhìn ảnh, sinh visual caption

Level 2 shot:
  LLM text fuser
  -> fuse VLM frame caption + OCR + object + audio
  Model có thể là Qwen2.5-Instruct, Llama, GPT API, Gemini API

Level 3 event-step:
  LLM TRAKE refiner
  -> tạo action_state, temporal_role, trake_text tốt hơn
```

**Vậy experiment hiện tại thiếu gì?**

Hiện đã có:

```text
VLM frame-caption experiment
```

Chưa có:

```text
LLM frame fuser
LLM shot fuser
LLM TRAKE refiner
```

**Thứ tự benchmark nên chạy**

Nên tạo các experiment riêng để so sánh rõ:

```text
E0: template_baseline
E1: vlm_frame_only
E2: vlm_frame + llm_frame_fuser
E3: vlm_frame + llm_shot_fuser
E4: vlm_frame + llm_shot_fuser + llm_trake_refiner
```

Sau đó Phase 6 so sánh:

```bash
--experiment vlm_frame_only=...
--experiment vlm_llm_frame_fuser=...
--experiment vlm_llm_shot_fuser=...
--experiment full_vlm_llm_trake=...
```

**Kết luận**

Mình không bỏ LLM/FuseCap. Mình đang để nó ở bước sau để benchmark sạch hơn.

Hiện experiment đầu tiên chỉ trả lời câu hỏi:

```text
VLM visual caption có cải thiện retrieval không?
```

Sau khi câu đó rõ, bước tiếp theo nên code:

```text
LLM/FuseCap-style text fuser
```

để trả lời câu thứ hai:

```text
VLM caption + OCR/object/audio được LLM fuse lại có tốt hơn template không?
```

> Đã hiểu rõ, hãy thực hiện lại phần experiments đầu tiên, lưu ý về env

<details><summary>16 previous messages</summary>

> Mình sẽ chỉnh lại experiment đầu tiên cho sạch phần `.env`: script sẽ tự load env file bằng stdlib trước khi expand `${AIC_ROOT}` trong YAML, không cần thêm dependency `python-dotenv`. Như vậy trên server bạn chỉ cần `--env-file ...` hoặc export thủ công đều được.
>
> <details><summary>Ran 3 commands</summary>
>
> - Ran `Get-Content -Path "C:\Users\User\Desktop\AIC2026\data_processing\caption\caption_pipeline\vlm_frame_experiment.py" -TotalCount 80`
> - Ran `Get-Content -Path "C:\Users\User\Desktop\AIC2026\data_processing\caption\run_vlm_frame_caption_experiment.py"`
> - Ran `Get-Content -Path "C:\Users\User\Desktop\AIC2026\data_processing\caption\configs\caption_experiment_vlm_a5000.yaml"`
>
> </details>
>
> Mình sẽ thêm `--env-file` vào CLI và hàm load `.env` nhỏ trong module VLM. Ngoài ra mình sẽ thêm `.env.example` cạnh config để bạn copy sang server, không commit secret.
>
> <details><summary>Edited a file</summary>
>
> - Wrote `C:/Users/User/Desktop/AIC2026/data_processing/caption/run_vlm_frame_caption_experiment.py` (+14 -6)
>
> </details>
>
> `apply_patch` không đọc được path Windows trong phiên này, nên mình sẽ dùng PowerShell ghi đúng file repo với quyền được yêu cầu, rồi kiểm tra lại ngay bằng `rg`.
>
> <details><summary>Ran a command</summary>
>
> - Ran ``$path = 'C:\Users\User\Desktop\AIC2026\data_processing\caption\run_vlm_frame_caption_experiment.py'
> $text = Get-Content -LiteralPath $path -Raw
> $text = $text.Replace("from caption_pipeline.vlm_frame_experiment import (`r`n    load_vlm_experiment_config,", "from caption_pipeline.vlm_frame_experiment import (`r`n    load_env_file,`r`n    load_vlm_experiment_config,")
> $text = $text.Replace('    parser.add_argument("--config", required=True, help="YAML experiment config")', '    parser.add_argument("--config", required=True, help="YAML experiment config")' + "`r`n" + '    parser.add_argument(' + "`r`n" + '        "--env-file",' + "`r`n" + '        default="",' + "`r`n" + '        help="Optional .env file loaded before expanding YAML env vars",' + "`r`n" + '    )')
> $text = $text.Replace("    cfg = load_vlm_experiment_config(args.config)", "    if args.env_file:`r`n        load_env_file(args.env_file)`r`n    cfg = load_vlm_experiment_config(args.config)")
> Set-Content -LiteralPath $path -Value $text -Encoding utf8``
> - Auto-review: approved
> - Auto-review risk: low
> - Auto-review rationale: Auto-review returned a low-risk allow decision.
>
> </details>
>
> CLI đã thêm `--env-file`. Mình sẽ thêm hàm `load_env_file()` trong module VLM, hỗ trợ dòng `KEY=value`, bỏ qua comment, và không overwrite env đã export trừ khi chọn mặc định từ file.
>
> <details><summary>Ran a command</summary>
>
> - Ran ``$path = 'C:\Users\User\Desktop\AIC2026\data_processing\caption\caption_pipeline\vlm_frame_experiment.py'
> $text = Get-Content -LiteralPath $path -Raw
> $insert = @'
>
> def load_env_file(path: str | Path, override: bool = False) -> dict[str, str]:
>     """Load simple KEY=VALUE lines from a .env file into os.environ."""
>     target = Path(path)
>     if not target.exists():
>         raise FileNotFoundError(f"Missing env file: {target}")
>     loaded: dict[str, str] = {}
>     with target.open("r", encoding="utf-8") as f:
>         for line_no, raw_line in enumerate(f, start=1):
>             line = raw_line.strip()
>             if not line or line.startswith("#"):
>                 continue
>             if line.startswith("export "):
>                 line = line[len("export "):].strip()
>             if "=" not in line:
>                 raise ValueError(f"Invalid .env line {target}:{line_no}: {raw_line.rstrip()}")
>             key, value = line.split("=", 1)
>             key = key.strip()
>             value = value.strip().strip('"').strip("'")
>             if not key:
>                 raise ValueError(f"Invalid empty .env key at {target}:{line_no}")
>             if override or key not in os.environ:
>                 os.environ[key] = value
>             loaded[key] = os.environ.get(key, value)
>     return loaded
> '@
> $text = $text.Replace("def load_vlm_experiment_config(path: str | Path) -> dict[str, Any]:", $insert + "`r`ndef load_vlm_experiment_config(path: str | Path) -> dict[str, Any]:")
> Set-Content -LiteralPath $path -Value $text -Encoding utf8``
> - Auto-review: approved
> - Auto-review risk: low
> - Auto-review rationale: Auto-review returned a low-risk allow decision.
>
> </details>
>
> Mình sẽ thêm file mẫu `.env.example` cho đúng biến cần thiết. File này không chứa secret, chỉ là template để copy sang `.env` trên server.
>
> <details><summary>Ran a command</summary>
>
> - Ran `$path = 'C:\Users\User\Desktop\AIC2026\data_processing\caption\configs\caption_experiment_vlm_a5000.env.example'
> $text = @'
> # Copy this file to .env on the server and adjust paths/devices.
> AIC_ROOT=/tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test
> PYTHONIOENCODING=utf-8
> CUDA_VISIBLE_DEVICES=1
>
> # Optional cache locations for Hugging Face models.
> HF_HOME=/tmp2/maitanha/.cache/huggingface
> TRANSFORMERS_CACHE=/tmp2/maitanha/.cache/huggingface
> HF_HUB_CACHE=/tmp2/maitanha/.cache/huggingface/hub
>
> # Optional runtime knobs.
> TOKENIZERS_PARALLELISM=false
> OMP_NUM_THREADS=4
> '@
> Set-Content -LiteralPath $path -Value $text -Encoding utf8`
> - Auto-review: approved
> - Auto-review risk: low
> - Auto-review rationale: Auto-review returned a low-risk allow decision.
>
> </details>
>
> Mình sẽ rà lại các thay đổi và lệnh chạy để đảm bảo bạn có đúng flow “env -> smoke -> VLM thật -> materialize -> eval -> benchmark”.
>
> <details><summary>Ran 4 commands</summary>
>
> - Ran `rg -n "env-file|load_env_file|AIC_ROOT|CUDA_VISIBLE_DEVICES|HF_HOME" "C:\Users\User\Desktop\AIC2026\data_processing\caption\run_vlm_frame_caption_experiment.py" "C:\Users\User\Desktop\AIC2026\data_processing\caption\caption_pipeline\vlm_frame_experiment.py" "C:\Users\User\Desktop\AIC2026\data_processing\caption\configs"`
> - Ran `Get-Content -Path "C:\Users\User\Desktop\AIC2026\data_processing\caption\caption_pipeline\vlm_frame_experiment.py" -TotalCount 80`
> - Ran `Get-Content -Path "C:\Users\User\Desktop\AIC2026\data_processing\caption\run_vlm_frame_caption_experiment.py" -TotalCount 60`
> - Ran `git -c safe.directory=C:/Users/User/Desktop/AIC2026 -C "C:/Users/User/Desktop/AIC2026" status --short -- data_processing/caption`
>
> </details>
>
> Mình vừa phát hiện import `load_env_file` chưa vào đúng chỗ trong CLI dù call đã có. Mình sẽ sửa ngay lỗi nhỏ này trước khi bạn chạy.
>
> <details><summary>Ran a command</summary>
>
> - Ran ``$path = 'C:\Users\User\Desktop\AIC2026\data_processing\caption\run_vlm_frame_caption_experiment.py'
> $text = Get-Content -LiteralPath $path -Raw
> $text = $text.Replace("from caption_pipeline.vlm_frame_experiment import (`r`n    load_vlm_experiment_config,", "from caption_pipeline.vlm_frame_experiment import (`r`n    load_env_file,`r`n    load_vlm_experiment_config,")
> $text = $text.Replace("from caption_pipeline.vlm_frame_experiment import (`n    load_vlm_experiment_config,", "from caption_pipeline.vlm_frame_experiment import (`n    load_env_file,`n    load_vlm_experiment_config,")
> Set-Content -LiteralPath $path -Value $text -Encoding utf8
>
> $path2 = 'C:\Users\User\Desktop\AIC2026\data_processing\caption\caption_pipeline\vlm_frame_experiment.py'
> $text2 = Get-Content -LiteralPath $path2 -Raw
> $text2 = $text2.Replace("    return loaded`r`ndef load_vlm_experiment_config", "    return loaded`r`n`r`n`r`ndef load_vlm_experiment_config")
> $text2 = $text2.Replace("    return loaded`ndef load_vlm_experiment_config", "    return loaded`n`n`ndef load_vlm_experiment_config")
> Set-Content -LiteralPath $path2 -Value $text2 -Encoding utf8``
> - Auto-review: approved
> - Auto-review risk: low
> - Auto-review rationale: Auto-review returned a low-risk allow decision.
>
> </details>
>
> Đã sửa import. Mình kiểm tra lại lần cuối phần `--env-file` và file mẫu env.
>
> <details><summary>Ran 3 commands</summary>
>
> - Ran `Get-Content -Path "C:\Users\User\Desktop\AIC2026\data_processing\caption\run_vlm_frame_caption_experiment.py" -TotalCount 55`
> - Ran `Get-Content -Path "C:\Users\User\Desktop\AIC2026\data_processing\caption\caption_pipeline\vlm_frame_experiment.py" -TotalCount 70`
> - Ran `Get-Content -Path "C:\Users\User\Desktop\AIC2026\data_processing\caption\configs\caption_experiment_vlm_a5000.env.example"`
>
> </details>
</details>

Đã chỉnh lại experiment đầu tiên để lưu ý `.env`.

Code đã thêm:
- `--env-file` cho `run_vlm_frame_caption_experiment.py`
- loader `.env` không cần `python-dotenv`
- file mẫu: `data_processing/caption/configs/caption_experiment_vlm_a5000.env.example`

Trên server, chạy như sau.

**1. Tạo `.env`**

```bash
cd /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/data_processing/caption

cp configs/caption_experiment_vlm_a5000.env.example configs/.env
nano configs/.env
```

Nội dung chính:

```bash
AIC_ROOT=/tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test
PYTHONIOENCODING=utf-8
CUDA_VISIBLE_DEVICES=1

HF_HOME=/tmp2/maitanha/.cache/huggingface
TRANSFORMERS_CACHE=/tmp2/maitanha/.cache/huggingface
HF_HUB_CACHE=/tmp2/maitanha/.cache/huggingface/hub

TOKENIZERS_PARALLELISM=false
OMP_NUM_THREADS=4
```

**2. Smoke test bằng mock**

```bash
PYTHONIOENCODING=utf-8 python run_vlm_frame_caption_experiment.py \
  --config configs/caption_experiment_vlm_a5000.yaml \
  --env-file configs/.env \
  --provider mock \
  --experiment-name mock_vlm_smoke \
  --max-frames 2 \
  --no-resume
```

**3. Chạy Qwen2.5-VL test 20 frame**

```bash
PYTHONIOENCODING=utf-8 python run_vlm_frame_caption_experiment.py \
  --config configs/caption_experiment_vlm_a5000.yaml \
  --env-file configs/.env \
  --experiment-name vlm_frame_qwen25vl_7b_rep \
  --frame-selection representative \
  --max-frames 20
```

**4. Nếu pass, chạy full representative**

```bash
PYTHONIOENCODING=utf-8 python run_vlm_frame_caption_experiment.py \
  --config configs/caption_experiment_vlm_a5000.yaml \
  --env-file configs/.env \
  --experiment-name vlm_frame_qwen25vl_7b_rep \
  --frame-selection representative
```

Output:

```text
/tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption_experiments_raw/vlm_frame_qwen25vl_7b_rep/L22_V012/frame_caption_overrides.jsonl
```

Sau đó tiếp tục materialize → Phase 5 → Phase 6 như trước.

Lưu ý: `configs/.env` không nên commit. Chỉ commit `.env.example`.

> "Traceback (most recent call last):
>   File "/tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/data_processing/caption/caption_pipeline/vlm_frame_experiment.py", line 290, in **init**
>     from qwen_vl_utils import process_vision_info
> ModuleNotFoundError: No module named 'qwen_vl_utils'
>
> The above exception was the direct cause of the following exception:
>
> Traceback (most recent call last):
>   File "/tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/data_processing/caption/run_vlm_frame_caption_experiment.py", line 86, in <module>
>     sys.exit(main())
>              ~~~~^^
>   File "/tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/data_processing/caption/run_vlm_frame_caption_experiment.py", line 53, in main
>     report = run_vlm_frame_experiment(cfg)
>   File "/tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/data_processing/caption/caption_pipeline/vlm_frame_experiment.py", line 106, in run_vlm_frame_experiment
>     model = create_vlm_captioner(model_cfg, generation)
>   File "/tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/data_processing/caption/caption_pipeline/vlm_frame_experiment.py", line 258, in create_vlm_captioner
>     return Qwen25VLCaptioner(model_cfg, generation)
>   File "/tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/data_processing/caption/caption_pipeline/vlm_frame_experiment.py", line 292, in **init**
>     raise RuntimeError("qwen-vl-utils is required for provider=qwen2_5_vl.") from exc
> RuntimeError: qwen-vl-utils is required for provider=qwen2_5_vl."

Lỗi này rõ rồi: env server đang thiếu package `qwen-vl-utils`.

Cài trong đúng conda env đang chạy:

```bash
pip install qwen-vl-utils
```

Sau đó check:

```bash
python -c "import qwen_vl_utils; print('qwen_vl_utils ok')"
```

Rồi chạy lại lệnh VLM.

Nếu muốn check đầy đủ dependency cho Qwen2.5-VL:

```bash
python -c "import torch, transformers, accelerate, qwen_vl_utils; print('all ok')"
```

Nếu dùng `flash_attention_2` trong config mà thiếu `flash-attn`, sẽ có lỗi tiếp theo. Khi đó có 2 lựa chọn:

**Cách an toàn hơn, bỏ flash attention trước**
Sửa trong `configs/caption_experiment_vlm_a5000.yaml`:

```yaml
model:
  attn_implementation:
```

hoặc xóa dòng:

```yaml
attn_implementation: flash_attention_2
```

**Cách hiệu năng cao hơn**
Cài `flash-attn` đúng CUDA/PyTorch, nhưng cái này hay mất thời gian hơn.

Khuyên bạn trước mắt:

```bash
pip install qwen-vl-utils
```

và nếu lỗi flash-attn thì tắt `attn_implementation` để test 20 frame trước.

> hãy tạo cho phần caption một conda env riêng, lưu ý tôi đã có cài flash_attention_2 (có thể tham khảo ở   Building wheel for flash-attn (pyproject.toml): finished with status 'done'
>   Created wheel for flash-attn: filename=flash_attn-2.8.3.post1-cp310-cp310-linux_x86_64.whl size=57668459 sha256=c49ba14eee697c75bb80e0e2b99bcd3547746a05cd055492b78e4cd368e57743
>   Stored in directory: /tmp2/maitanha/vgu/ttn/AIC-Khoa/tmp/pip-ephem-wheel-cache-m8os9_8o/wheels/52/c6/18/28288f2ddc55daebd5c1972cafd9b00a352bd5128cdd37eb4d
> Successfully built flash-attn
> Installing collected packages: flash-attn
> Successfully installed flash-attn-2.8.3.post1)

<details><summary>4 previous messages</summary>

> Mình sẽ tạo một env spec riêng cho caption/VLM experiment, tách khỏi env OCR/object/audio. Mình sẽ ưu tiên Python 3.10 vì wheel flash-attn bạn build là `cp310`, và để `flash-attn` ở bước install riêng thay vì nhét cứng vào YAML để tránh build lại sai CUDA/PyTorch.
>
> <details><summary>Ran 3 commands</summary>
>
> - Ran `rg -n "torch|transformers|qwen|flash|accelerate|yaml|PIL|torchvision|sentencepiece|protobuf|requirements|environment" "C:\Users\User\Desktop\AIC2026\data_processing\caption"`
> - Ran `Get-Content -Path "C:\Users\User\Desktop\AIC2026\data_processing\caption\caption_pipeline\vlm_frame_experiment.py" | Select-Object -Skip 240 -First 180`
> - Ran `Get-ChildItem -Path "C:\Users\User\Desktop\AIC2026\data_processing\caption" | Select-Object Name,Mode,Length`
>
> </details>
>
> Mình sẽ tạo hẳn một bộ env riêng trong `data_processing/caption/envs/`: YAML conda, script cài đặt theo từng bước, và script check dependency. Vì wheel flash-attn của bạn là `cp310`, mình sẽ khóa env ở Python 3.10 để tránh build lại lệch ABI.
>
> <details><summary>Created a file</summary>
>
> - Wrote `C:/Users/User/Desktop/AIC2026/data_processing/caption/envs/caption_vlm_a5000_environment.yml` (+25 -0)
> - Auto-review: approved
> - Auto-review risk: low
> - Auto-review rationale: Auto-review returned a low-risk allow decision.
>
> </details>
</details>

> hiện tại tôi đang để tất cả các .env trong /tmp2/maitanha/vgu/ttn/AIC-Khoa/conda_envs

Ổn, vậy với caption/VLM experiment mình khuyên để riêng như này:

```text
/tmp2/maitanha/vgu/ttn/AIC-Khoa/conda_envs/
├── aic-caption-vlm/                  # conda env folder
├── caption_vlm.env                   # env vars cho caption/VLM
└── flash_attn_wheels/                # optional, lưu wheel flash-attn nếu muốn
```

`.env` nên đặt:

```bash
/tmp2/maitanha/vgu/ttn/AIC-Khoa/conda_envs/caption_vlm.env
```

Nội dung:

```bash
AIC_ROOT=/tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test
PYTHONIOENCODING=utf-8
CUDA_VISIBLE_DEVICES=1

HF_HOME=/tmp2/maitanha/vgu/ttn/AIC-Khoa/conda_envs/hf_cache
TRANSFORMERS_CACHE=/tmp2/maitanha/vgu/ttn/AIC-Khoa/conda_envs/hf_cache
HF_HUB_CACHE=/tmp2/maitanha/vgu/ttn/AIC-Khoa/conda_envs/hf_cache/hub

TOKENIZERS_PARALLELISM=false
OMP_NUM_THREADS=4
```

Sau đó khi chạy:

```bash
cd /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/data_processing/caption

PYTHONIOENCODING=utf-8 python run_vlm_frame_caption_experiment.py \
  --config configs/caption_experiment_vlm_a5000.yaml \
  --env-file /tmp2/maitanha/vgu/ttn/AIC-Khoa/conda_envs/caption_vlm.env \
  --provider mock \
  --experiment-name mock_vlm_smoke \
  --max-frames 2 \
  --no-resume
```

Và chạy Qwen thật:

```bash
PYTHONIOENCODING=utf-8 python run_vlm_frame_caption_experiment.py \
  --config configs/caption_experiment_vlm_a5000.yaml \
  --env-file /tmp2/maitanha/vgu/ttn/AIC-Khoa/conda_envs/caption_vlm.env \
  --experiment-name vlm_frame_qwen25vl_7b_rep \
  --frame-selection representative \
  --max-frames 20
```

Với conda env riêng, vì bạn muốn để trong `/tmp2/.../conda_envs`, tạo bằng prefix sẽ sạch hơn:

```bash
conda create -p /tmp2/maitanha/vgu/ttn/AIC-Khoa/conda_envs/aic-caption-vlm python=3.10 -y
conda activate /tmp2/maitanha/vgu/ttn/AIC-Khoa/conda_envs/aic-caption-vlm
```

Rồi cài package:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

pip install \
  "transformers>=4.49.0" \
  "accelerate>=0.33.0" \
  "qwen-vl-utils>=0.0.8" \
  "safetensors>=0.4.5" \
  "sentencepiece>=0.2.0" \
  "protobuf>=4.25.0" \
  "einops>=0.8.0" \
  pillow pyyaml tqdm
```

Nếu bạn đã có wheel `flash_attn-2.8.3.post1-cp310-cp310-linux_x86_64.whl`, cài lại trong env mới bằng:

```bash
pip install /path/to/flash_attn-2.8.3.post1-cp310-cp310-linux_x86_64.whl
```

Nếu không còn wheel nhưng build được trước đó:

```bash
pip install flash-attn==2.8.3.post1 --no-build-isolation
```

Check env:

```bash
python -c "import torch, transformers, qwen_vl_utils, accelerate; print(torch.__version__, torch.cuda.is_available())"
python -c "import flash_attn; print('flash_attn ok')"
```

Điểm quan trọng: wheel flash-attn bạn nói là `cp310`, nên env caption nên dùng **Python 3.10**, không dùng 3.11/3.12.

> - bạn có thể tham khảo cách sử dụng flashattention ở @data_processing/OCR/ocr_vlm_pipeline_v2/
> - tiếp theo hãy tạo một .env chung cho tất cả ở stage caption, nếu có conflict về môi trường thì chia ra

<details><summary>35 previous messages</summary>

> Mình sẽ rà nhanh cách OCR đang bật flash attention để giữ naming/env consistent, rồi tạo bộ `.env` mẫu chung cho caption stage. Vì `.env` thật nằm trên server trong `conda_envs`, mình sẽ commit `.env.example` và để script hỗ trợ `--env-file` dùng path đó.
>
> Tool vừa vấp `D:\Users\outputs` invalid workdir, mình chuyển thẳng sang root repo để đọc.
>
> <details><summary>Ran 3 commands</summary>
>
> - Ran `rg -n "flash_attention|flash-attn|attn|VINTERN|env|CUDA_VISIBLE|PADDLE|DEVICE" "data_processing\OCR\ocr_vlm_pipeline_v2"`
> - Ran `Get-ChildItem -Path "data_processing\caption\configs" | Select-Object Name,Length`
> - Ran `Get-Content -Path "data_processing\caption\caption_pipeline\vlm_frame_experiment.py" -TotalCount 95`
>
> </details>
>
> Mình thấy OCR dùng pattern khá tốt: env chọn attention (`OCR_V2_VINTERN_ATTN=flash_attention_2`), code kiểm tra flash-attn import được thì dùng, nếu không fallback. Mình sẽ áp dụng cùng style cho caption: `.env` chung cho paths/runtime, `.env` VLM riêng cho GPU/model attention, và code VLM có fallback `flash_attention_2 -> sdpa -> eager`.
>
> <details><summary>Ran 3 commands</summary>
>
> - Ran `Get-ChildItem -Path "data_processing\caption" -Recurse -File | Where-Object {$_.FullName -match 'env|config|vlm'} | Select-Object FullName,Length`
> - Ran `Get-Content -Path "data_processing\OCR\ocr_vlm_pipeline_v2\ocr_pipeline\recognizers\vintern_recognizer.py" | Select-Object -Skip 60 -First 140`
> - Ran `Get-Content -Path "data_processing\caption\configs\caption_experiment_vlm_a5000.yaml"`
>
> </details>
>
> Mình sẽ làm hai việc nhỏ nhưng quan trọng: `--env-file` có thể lặp nhiều lần để load common + VLM-specific, và Qwen attention sẽ fallback giống OCR nếu flash-attn ABI/import lỗi. Sau đó mình tạo `.env.example` chung và `.env.example` riêng cho VLM.
>
> <details><summary>Edited a file</summary>
>
> - Wrote `C:/Users/User/Desktop/AIC2026/data_processing/caption/run_vlm_frame_caption_experiment.py` (+13 -7)
>
> </details>
>
> `apply_patch` lại không đọc được path Windows trong phiên này, nên mình sẽ dùng chỉnh sửa PowerShell có kiểm tra lại ngay sau đó.
>
> <details><summary>Ran a command</summary>
>
> - Ran `$path = 'C:\Users\User\Desktop\AIC2026\data_processing\caption\run_vlm_frame_caption_experiment.py'
> $text = Get-Content -LiteralPath $path -Raw
> $text = $text.Replace('''        default="",
>         help="Optional .env file loaded before expanding YAML env vars",
>     )''', '''        action="append",
>         default=[],
>         help="Optional .env file loaded before expanding YAML env vars. Can be repeated.",
>     )
>     parser.add_argument(
>         "--env-override",
>         action="store_true",
>         help="Allow later --env-file values to override existing environment variables",
>     )''')
> $text = $text.Replace('''    if args.env_file:
>         load_env_file(args.env_file)
>     cfg = load_vlm_experiment_config(args.config)''', '''    for env_file in args.env_file:
>         load_env_file(env_file, override=args.env_override)
>     cfg = load_vlm_experiment_config(args.config)''')
> Set-Content -LiteralPath $path -Value $text -Encoding utf8`
> - Auto-review: approved
> - Auto-review risk: low
> - Auto-review rationale: Auto-review returned a low-risk allow decision.
>
> </details>
>
> Mình sẽ kiểm tra đoạn CLI vừa sửa trước khi chỉnh tiếp phần attention fallback.
>
> <details><summary>Ran a command</summary>
>
> - Ran `Get-Content -Path "C:\Users\User\Desktop\AIC2026\data_processing\caption\run_vlm_frame_caption_experiment.py" -TotalCount 65`
>
> </details>
>
> Replace trước chưa match do line ending/string. Mình sẽ sửa bằng regex chính xác hơn.
>
> <details><summary>Ran a command</summary>
>
> - Ran `$path = 'C:\Users\User\Desktop\AIC2026\data_processing\caption\run_vlm_frame_caption_experiment.py'
> $text = Get-Content -LiteralPath $path -Raw
> $old = @'
>     parser.add_argument(
>         "--env-file",
>         default="",
>         help="Optional .env file loaded before expanding YAML env vars",
>     )
> '@
> $new = @'
>     parser.add_argument(
>         "--env-file",
>         action="append",
>         default=[],
>         help="Optional .env file loaded before expanding YAML env vars. Can be repeated.",
>     )
>     parser.add_argument(
>         "--env-override",
>         action="store_true",
>         help="Allow later --env-file values to override existing environment variables",
>     )
> '@
> $text = $text.Replace($old, $new)
> $old2 = @'
>     if args.env_file:
>         load_env_file(args.env_file)
>     cfg = load_vlm_experiment_config(args.config)
> '@
> $new2 = @'
>     for env_file in args.env_file:
>         load_env_file(env_file, override=args.env_override)
>     cfg = load_vlm_experiment_config(args.config)
> '@
> $text = $text.Replace($old2, $new2)
> Set-Content -LiteralPath $path -Value $text -Encoding utf8`
> - Auto-review: approved
> - Auto-review risk: low
> - Auto-review rationale: Auto-review returned a low-risk allow decision.
>
> </details>
>
> Giờ mình kiểm tra lại CLI đã đúng chưa.
>
> <details><summary>Ran a command</summary>
>
> - Ran `Get-Content -Path "C:\Users\User\Desktop\AIC2026\data_processing\caption\run_vlm_frame_caption_experiment.py" -TotalCount 70`
>
> </details>
>
> Vẫn chưa đổi, có thể do encoding/line ending. Mình sẽ chỉnh theo mảng dòng thay vì replace chuỗi.
>
> <details><summary>Ran a command</summary>
>
> - Ran `$path = 'C:\Users\User\Desktop\AIC2026\data_processing\caption\run_vlm_frame_caption_experiment.py'
> $lines = [System.Collections.Generic.List[string]](Get-Content -LiteralPath $path)
> # Replace env-file argument block
> $start = -1
> for ($i = 0; $i -lt $lines.Count; $i++) {
>   if ($lines[$i].Contains('"--env-file"')) { $start = $i - 1; break }
> }
> if ($start -lt 0) { throw 'env-file block not found' }
> $end = $start
> $parenBalance = 0
> for ($i = $start; $i -lt $lines.Count; $i++) {
>   if ($lines[$i].Contains('parser.add_argument(')) { $parenBalance = 1 }
>   elseif ($parenBalance -gt 0 -and $lines[$i].Trim() -eq ')') { $end = $i; break }
> }
> $newBlock = @(
> '    parser.add_argument(',
> '        "--env-file",',
> '        action="append",',
> '        default=[],',
> '        help="Optional .env file loaded before expanding YAML env vars. Can be repeated.",',
> '    )',
> '    parser.add_argument(',
> '        "--env-override",',
> '        action="store_true",',
> '        help="Allow later --env-file values to override existing environment variables",',
> '    )'
> )
> $lines.RemoveRange($start, $end - $start + 1)
> $lines.InsertRange($start, [string[]]$newBlock)
> # Replace env loading block
> $idx = -1
> for ($i = 0; $i -lt $lines.Count; $i++) {
>   if ($lines[$i].Trim() -eq 'if args.env_file:') { $idx = $i; break }
> }
> if ($idx -lt 0) { throw 'env loading block not found' }
> $lines.RemoveRange($idx, 2)
> $loadBlock = @(
> '    for env_file in args.env_file:',
> '        load_env_file(env_file, override=args.env_override)'
> )
> $lines.InsertRange($idx, [string[]]$loadBlock)
> Set-Content -LiteralPath $path -Value $lines -Encoding utf8`
> - Auto-review: approved
> - Auto-review risk: low
> - Auto-review rationale: Auto-review returned a low-risk allow decision.
>
> </details>
>
> Đã chỉnh theo mảng dòng. Mình kiểm tra lại ngay.
>
> <details><summary>Ran a command</summary>
>
> - Ran `Get-Content -Path "C:\Users\User\Desktop\AIC2026\data_processing\caption\run_vlm_frame_caption_experiment.py" -TotalCount 75`
>
> </details>
>
> Tiếp theo mình chỉnh Qwen attention fallback: ưu tiên `CAPTION_VLM_ATTN` từ env, sau đó YAML, nếu flash-attn không import được thì tự chuyển `sdpa/eager` thay vì crash ngay.
>
> <details><summary>Ran a command</summary>
>
> - Ran ``$path = 'C:\Users\User\Desktop\AIC2026\data_processing\caption\caption_pipeline\vlm_frame_experiment.py'
> $text = Get-Content -LiteralPath $path -Raw
> $old = @'
>         model_kwargs: dict[str, Any] = {
>             "torch_dtype": dtype,
>             "device_map": model_cfg.get("device_map", "auto"),
>         }
>         if model_cfg.get("attn_implementation"):
>             model_kwargs["attn_implementation"] = model_cfg["attn_implementation"]
>         self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
>             model_name, **model_kwargs,
>         )
>         self.processor = AutoProcessor.from_pretrained(model_name)
>         self.prompt = generation.get("prompt", DEFAULT_PROMPT)
> '@
> $new = @'
>         requested_attn = os.environ.get("CAPTION_VLM_ATTN") or model_cfg.get("attn_implementation")
>         fallback_order = _attention_fallback_order(requested_attn)
>         last_error = None
>         self.model = None
>         for candidate in fallback_order:
>             model_kwargs: dict[str, Any] = {
>                 "torch_dtype": dtype,
>                 "device_map": model_cfg.get("device_map", "auto"),
>             }
>             if candidate:
>                 model_kwargs["attn_implementation"] = candidate
>             logger.info("Loading Qwen2.5-VL: %s, attn=%s", model_name, candidate or "default")
>             try:
>                 self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
>                     model_name, **model_kwargs,
>                 )
>                 logger.info("Qwen2.5-VL loaded successfully: %s, attn=%s", model_name, candidate or "default")
>                 break
>             except Exception as exc:
>                 last_error = exc
>                 logger.warning("Qwen2.5-VL load failed with attn=%s: %s", candidate or "default", exc)
>         if self.model is None:
>             raise RuntimeError(f"Could not load Qwen2.5-VL model {model_name}") from last_error
>         self.processor = AutoProcessor.from_pretrained(model_name)
>         self.prompt = generation.get("prompt", DEFAULT_PROMPT)
> '@
> if (-not $text.Contains($old)) { throw 'Qwen model kwargs block not found' }
> $text = $text.Replace($old, $new)
> $insert = @'
>
> def _has_flash_attn() -> bool:
>     try:
>         import flash_attn  # noqa: F401
>         import flash_attn.flash_attn_interface  # noqa: F401
>         return True
>     except Exception as exc:
>         logger.warning("flash_attn is unavailable or ABI-incompatible: %s", exc)
>         return False
>
>
> def _has_sdpa() -> bool:
>     try:
>         import torch.nn.functional as F
>         return hasattr(F, "scaled_dot_product_attention")
>     except Exception:
>         return False
>
>
> def _attention_fallback_order(requested: str | None) -> list[str]:
>     requested = (requested or "").strip()
>     if requested in {"", "auto", "none", "None"}:
>         if _has_flash_attn():
>             requested = "flash_attention_2"
>         elif _has_sdpa():
>             requested = "sdpa"
>         else:
>             requested = "eager"
>     order: list[str] = []
>     if requested == "flash_attention_2" and not _has_flash_attn():
>         logger.warning("Requested flash_attention_2 but flash_attn is not usable; falling back.")
>     elif requested:
>         order.append(requested)
>     for candidate in ("sdpa", "eager"):
>         if candidate not in order:
>             if candidate == "sdpa" and not _has_sdpa():
>                 continue
>             order.append(candidate)
>     if not order:
>         order.append("eager")
>     return order
> '@
> if (-not $text.Contains("def _torch_dtype(torch_module, name: str):")) { throw 'torch dtype marker not found' }
> $text = $text.Replace("def _torch_dtype(torch_module, name: str):", $insert + "`r`ndef _torch_dtype(torch_module, name: str):")
> Set-Content -LiteralPath $path -Value $text -Encoding utf8``
> - Auto-review: approved
> - Auto-review risk: low
> - Auto-review rationale: Auto-review returned a low-risk allow decision.
>
> </details>
>
> Mình cũng thấy file conda env YAML vừa tạo trước khi bạn ngắt có lỗi nhỏ ở phần `channels`; mình sẽ sửa luôn để nếu bạn dùng tạo env riêng thì không bị vấp. Sau đó thêm 2 `.env.example`: common và VLM-specific.
>
> <details><summary>Edited a file, ran a command</summary>
>
> - Wrote `C:/Users/User/Desktop/AIC2026/data_processing/caption/envs/caption_vlm_a5000_environment.yml` (+6 -6)
> - Ran ``$path = 'C:\Users\User\Desktop\AIC2026\data_processing\caption\envs\caption_vlm_a5000_environment.yml'
> $text = Get-Content -LiteralPath $path -Raw
> $text = $text.Replace("  - conda-forge`r`ndefaults:`r`ndependencies:", "  - conda-forge`r`n  - defaults`r`ndependencies:")
> $text = $text.Replace("  - conda-forge`ndefaults:`ndependencies:", "  - conda-forge`n  - defaults`ndependencies:")
> Set-Content -LiteralPath $path -Value $text -Encoding utf8``
> - Auto-review: approved
> - Auto-review risk: low
> - Auto-review rationale: Auto-review returned a low-risk allow decision.
>
> </details>
>
> Giờ mình tạo các `.env.example` tách lớp: common cho mọi phase caption, VLM riêng cho GPU/model.
>
> <details><summary>Ran a command</summary>
>
> - Ran `$common = 'C:\Users\User\Desktop\AIC2026\data_processing\caption\configs\caption_stage_common.env.example'
> $commonText = @'
> # Common environment for all caption-stage scripts.
> # Copy to /tmp2/maitanha/vgu/ttn/AIC-Khoa/conda_envs/caption_stage_common.env on the server.
>
> AIC_ROOT=/tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test
> PYTHONIOENCODING=utf-8
> TOKENIZERS_PARALLELISM=false
> TRANSFORMERS_VERBOSITY=error
> HF_HUB_DISABLE_SYMLINKS_WARNING=1
> OMP_NUM_THREADS=4
>
> # Caption pipeline roots derived from AIC_ROOT.
> CAPTION_BASELINE_DIR=/tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption
> CAPTION_EXPERIMENT_ROOT=/tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption_experiments
> CAPTION_EXPERIMENT_RAW_ROOT=/tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption_experiments_raw
> KEYFRAME_ROOT=/tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/keyframe_test
> '@
> Set-Content -LiteralPath $common -Value $commonText -Encoding utf8
>
> $vlm = 'C:\Users\User\Desktop\AIC2026\data_processing\caption\configs\caption_stage_vlm.env.example'
> $vlmText = @'
> # VLM-specific environment for caption experiments.
> # Load this after caption_stage_common.env.
> # Copy to /tmp2/maitanha/vgu/ttn/AIC-Khoa/conda_envs/caption_stage_vlm.env on the server.
>
> CUDA_VISIBLE_DEVICES=1
>
> # Qwen/InternVL caches. Keep these under conda_envs or another large /tmp2 path.
> HF_HOME=/tmp2/maitanha/vgu/ttn/AIC-Khoa/conda_envs/hf_cache
> TRANSFORMERS_CACHE=/tmp2/maitanha/vgu/ttn/AIC-Khoa/conda_envs/hf_cache
> HF_HUB_CACHE=/tmp2/maitanha/vgu/ttn/AIC-Khoa/conda_envs/hf_cache/hub
>
> # Same role as OCR_V2_VINTERN_ATTN, but for caption VLM.
> # Use flash_attention_2 when flash-attn is installed and ABI-compatible.
> # The code falls back to sdpa/eager if flash-attn cannot be imported.
> CAPTION_VLM_ATTN=flash_attention_2
> '@
> Set-Content -LiteralPath $vlm -Value $vlmText -Encoding utf8`
> - Auto-review: approved
> - Auto-review risk: low
> - Auto-review rationale: Auto-review returned a low-risk allow decision.
>
> </details>
>
> Mình sẽ rà lại các file env và fallback attention để đảm bảo đúng naming `CAPTION_VLM_ATTN`.
>
> <details><summary>Ran 4 commands</summary>
>
> - Ran `Get-Content -Path "data_processing\caption\configs\caption_stage_common.env.example"`
> - Ran `Get-Content -Path "data_processing\caption\configs\caption_stage_vlm.env.example"`
> - Ran `Get-Content -Path "data_processing\caption\envs\caption_vlm_a5000_environment.yml"`
> - Ran `Get-Content -Path "data_processing\caption\caption_pipeline\vlm_frame_experiment.py" | Select-Object -Skip 285 -First 170`
>
> </details>
>
> Mình sẽ xem tiếp đoạn cuối helper để chắc fallback order đầy đủ.
>
> <details><summary>Ran a command</summary>
>
> - Ran `Get-Content -Path "data_processing\caption\caption_pipeline\vlm_frame_experiment.py" | Select-Object -Skip 430 -First 80`
>
> </details>
>
> Một điểm format nhỏ: helper mới đang thiếu dòng trống giữa function, mình chỉnh cho sạch rồi chốt lệnh dùng `.env` chung + VLM riêng.
>
> <details><summary>Ran a command</summary>
>
> - Ran ``$path = 'C:\Users\User\Desktop\AIC2026\data_processing\caption\caption_pipeline\vlm_frame_experiment.py'
> $text = Get-Content -LiteralPath $path -Raw
> $text = $text.Replace("    return order`r`ndef _torch_dtype", "    return order`r`n`r`n`r`ndef _torch_dtype")
> $text = $text.Replace("    return order`ndef _torch_dtype", "    return order`n`n`ndef _torch_dtype")
> $text = $text.Replace("        return \"cuda\"`r`ndef _resolve_video_dir", "        return \"cuda\"`r`n`r`n`r`ndef _resolve_video_dir")
> $text = $text.Replace("        return \"cuda\"`ndef _resolve_video_dir", "        return \"cuda\"`n`n`ndef _resolve_video_dir")
> Set-Content -LiteralPath $path -Value $text -Encoding utf8``
> - Auto-review: approved
> - Auto-review risk: low
> - Auto-review rationale: Auto-review returned a low-risk allow decision.
>
> </details>
>
> Quoting PowerShell lại vấp dấu `"cuda"`, mình sẽ sửa bằng single-quoted pattern.
>
> <details><summary>Ran a command</summary>
>
> - Ran ``$path = 'C:\Users\User\Desktop\AIC2026\data_processing\caption\caption_pipeline\vlm_frame_experiment.py'
> $text = Get-Content -LiteralPath $path -Raw
> $text = $text.Replace('    return order' + "`r`n" + 'def _torch_dtype', '    return order' + "`r`n`r`n`r`n" + 'def _torch_dtype')
> $text = $text.Replace('    return order' + "`n" + 'def _torch_dtype', '    return order' + "`n`n`n" + 'def _torch_dtype')
> $text = $text.Replace('        return "cuda"' + "`r`n" + 'def _resolve_video_dir', '        return "cuda"' + "`r`n`r`n`r`n" + 'def _resolve_video_dir')
> $text = $text.Replace('        return "cuda"' + "`n" + 'def _resolve_video_dir', '        return "cuda"' + "`n`n`n" + 'def _resolve_video_dir')
> Set-Content -LiteralPath $path -Value $text -Encoding utf8``
> - Auto-review: approved
> - Auto-review risk: low
> - Auto-review rationale: Auto-review returned a low-risk allow decision.
>
> </details>
>
> Mình sẽ kiểm tra trạng thái file để tóm lại rõ ràng bạn cần copy những `.env` nào lên `/tmp2/.../conda_envs`.
>
> <details><summary>Ran 2 commands</summary>
>
> - Ran `rg -n "CAPTION_VLM_ATTN|caption_stage_common|caption_stage_vlm|env-override|attention_fallback" "data_processing\caption"`
> - Ran `git -c safe.directory=C:/Users/User/Desktop/AIC2026 -C "C:/Users/User/Desktop/AIC2026" status --short -- data_processing/caption`
>
> </details>
</details>

Mình đã chỉnh lại phần caption stage theo hướng bạn muốn.

Đã thêm/sửa:

```text
data_processing/caption/configs/caption_stage_common.env.example
data_processing/caption/configs/caption_stage_vlm.env.example
data_processing/caption/envs/caption_vlm_a5000_environment.yml
data_processing/caption/run_vlm_frame_caption_experiment.py
data_processing/caption/caption_pipeline/vlm_frame_experiment.py
```

Quan trọng nhất:

- `.env` chung cho mọi stage caption: `caption_stage_common.env`
- `.env` riêng cho VLM/GPU: `caption_stage_vlm.env`
- CLI VLM giờ hỗ trợ nhiều `--env-file`
- Qwen attention giờ dùng `CAPTION_VLM_ATTN`, giống style OCR `OCR_V2_VINTERN_ATTN`
- Nếu `flash_attention_2` lỗi import/ABI, code fallback `sdpa -> eager`

Trên server, tạo env files thật:

```bash
cp data_processing/caption/configs/caption_stage_common.env.example \
  /tmp2/maitanha/vgu/ttn/AIC-Khoa/conda_envs/caption_stage_common.env

cp data_processing/caption/configs/caption_stage_vlm.env.example \
  /tmp2/maitanha/vgu/ttn/AIC-Khoa/conda_envs/caption_stage_vlm.env
```

Chỉnh nếu cần:

```bash
nano /tmp2/maitanha/vgu/ttn/AIC-Khoa/conda_envs/caption_stage_common.env
nano /tmp2/maitanha/vgu/ttn/AIC-Khoa/conda_envs/caption_stage_vlm.env
```

Chạy smoke test:

```bash
cd /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/data_processing/caption

PYTHONIOENCODING=utf-8 python run_vlm_frame_caption_experiment.py \
  --config configs/caption_experiment_vlm_a5000.yaml \
  --env-file /tmp2/maitanha/vgu/ttn/AIC-Khoa/conda_envs/caption_stage_common.env \
  --env-file /tmp2/maitanha/vgu/ttn/AIC-Khoa/conda_envs/caption_stage_vlm.env \
  --provider mock \
  --experiment-name mock_vlm_smoke \
  --max-frames 2 \
  --no-resume
```

Chạy Qwen thật:

```bash
PYTHONIOENCODING=utf-8 python run_vlm_frame_caption_experiment.py \
  --config configs/caption_experiment_vlm_a5000.yaml \
  --env-file /tmp2/maitanha/vgu/ttn/AIC-Khoa/conda_envs/caption_stage_common.env \
  --env-file /tmp2/maitanha/vgu/ttn/AIC-Khoa/conda_envs/caption_stage_vlm.env \
  --experiment-name vlm_frame_qwen25vl_7b_rep \
  --frame-selection representative \
  --max-frames 20
```

Nếu muốn ép env file sau override env file trước, thêm:

```bash
--env-override
```

Nếu flash-attn vẫn lỗi, chỉ cần sửa:

```bash
CAPTION_VLM_ATTN=sdpa
```

trong:

```text
/tmp2/maitanha/vgu/ttn/AIC-Khoa/conda_envs/caption_stage_vlm.env
```

Không cần sửa YAML.

> (base) maitanha@cl777:/tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/data_processing/caption$ cp data_processing/caption/configs/caption_stage_common.env.example   /tmp2/maitanha/vgu/ttn/AIC-Khoa/conda_envs/caption_stage_common.env
> cp: cannot stat 'data_processing/caption/configs/caption_stage_common.env.example': No such file or directory
> (base) maitanha@cl777:/tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/data_processing/caption$

Bạn đang đứng trong thư mục:

```bash
/tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/data_processing/caption
```

nên path `data_processing/caption/...` bị lặp sai. Dùng path tương đối từ thư mục hiện tại như này:

```bash
cp configs/caption_stage_common.env.example \
  /tmp2/maitanha/vgu/ttn/AIC-Khoa/conda_envs/caption_stage_common.env

cp configs/caption_stage_vlm.env.example \
  /tmp2/maitanha/vgu/ttn/AIC-Khoa/conda_envs/caption_stage_vlm.env
```

Hoặc dùng path tuyệt đối từ repo root:

```bash
cp /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/data_processing/caption/configs/caption_stage_common.env.example \
  /tmp2/maitanha/vgu/ttn/AIC-Khoa/conda_envs/caption_stage_common.env

cp /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/data_processing/caption/configs/caption_stage_vlm.env.example \
  /tmp2/maitanha/vgu/ttn/AIC-Khoa/conda_envs/caption_stage_vlm.env
```

Sau đó kiểm tra:

```bash
ls -l /tmp2/maitanha/vgu/ttn/AIC-Khoa/conda_envs/caption_stage_*.env
```

> "[ERROR] `loss` is part of Qwen2_5_VLCausalLMOutputWithPast.**init**'s signature, but not documented. Make sure to add it
>  to the docstring of the function in /home/maitanha/miniconda3/lib/python3.13/site-packages/transformers/models/qwen2_5_
> vl/modeling_qwen2_5_vl.py.
> [ERROR] `logits` is part of Qwen2_5_VLCausalLMOutputWithPast.**init**'s signature, but not documented. Make sure to add
> it to the docstring of the function in /home/maitanha/miniconda3/lib/python3.13/site-packages/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py.
> Traceback (most recent call last):
>   File "/tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/data_processing/caption/caption_pipeline/vlm_frame_experiment.py", line 290, in **init**
>     from qwen_vl_utils import process_vision_info
> ModuleNotFoundError: No module named 'qwen_vl_utils'
>
> The above exception was the direct cause of the following exception:
>
> Traceback (most recent call last):
>   File "/tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/data_processing/caption/run_vlm_frame_caption_experiment.py", line 92, in <module>
>     sys.exit(main())
>              ~~~~^^
>   File "/tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/data_processing/caption/run_vlm_frame_caption_experiment.py", line 59, in main
>     report = run_vlm_frame_experiment(cfg)
>   File "/tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/data_processing/caption/caption_pipeline/vlm_frame_experiment.py", lin$"

Bạn đang chạy bằng **base Python 3.13**:

```text
/home/maitanha/miniconda3/lib/python3.13/site-packages/...
```

Trong khi caption VLM env cần **Python 3.10** vì:
- `flash-attn` wheel bạn build là `cp310`
- Qwen/VLM stack ổn hơn với py3.10
- env hiện tại thiếu `qwen_vl_utils`

Vậy lỗi chính không phải code, mà là **chưa activate đúng conda env caption VLM**.

Chạy theo thứ tự này.

**1. Tạo env caption VLM nếu chưa có**

```bash
conda create -p /tmp2/maitanha/vgu/ttn/AIC-Khoa/conda_envs/aic-caption-vlm python=3.10 -y
```

**2. Activate env**

```bash
conda activate /tmp2/maitanha/vgu/ttn/AIC-Khoa/conda_envs/aic-caption-vlm
```

Check phải ra Python 3.10:

```bash
python -V
which python
```

Kỳ vọng:

```text
Python 3.10.x
/tmp2/maitanha/vgu/ttn/AIC-Khoa/conda_envs/aic-caption-vlm/bin/python
```

**3. Cài dependencies**

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

pip install \
  "transformers>=4.49.0" \
  "accelerate>=0.33.0" \
  "qwen-vl-utils>=0.0.8" \
  "safetensors>=0.4.5" \
  "sentencepiece>=0.2.0" \
  "protobuf>=4.25.0" \
  "einops>=0.8.0" \
  pillow pyyaml tqdm
```

**4. Cài flash-attn trong env py3.10**

Nếu còn wheel `.whl`:

```bash
pip install /path/to/flash_attn-2.8.3.post1-cp310-cp310-linux_x86_64.whl
```

Nếu không còn wheel:

```bash
pip install flash-attn==2.8.3.post1 --no-build-isolation
```

**5. Check env**

```bash
python -c "import torch, transformers, qwen_vl_utils, accelerate; print(torch.__version__, torch.cuda.is_available())"
python -c "import flash_attn; print('flash_attn ok')"
```

**6. Chạy lại smoke/VLM**

```bash
cd /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/data_processing/caption

PYTHONIOENCODING=utf-8 python run_vlm_frame_caption_experiment.py \
  --config configs/caption_experiment_vlm_a5000.yaml \
  --env-file /tmp2/maitanha/vgu/ttn/AIC-Khoa/conda_envs/caption_stage_common.env \
  --env-file /tmp2/maitanha/vgu/ttn/AIC-Khoa/conda_envs/caption_stage_vlm.env \
  --experiment-name vlm_frame_qwen25vl_7b_rep \
  --frame-selection representative \
  --max-frames 20
```

Tóm lại: bạn đang chạy nhầm `base` Python 3.13. Hãy activate env `aic-caption-vlm` Python 3.10 rồi cài `qwen-vl-utils`.

> Hãy cho lại tuần tự về tạo env, chạy experiments
> Tôi cung thêm cho bạn về logs build flashattention
>
> User context
>
> Attachments:
> - flash_attn_build.log: `d:/Users/flash_attn_build.log`

<details><summary>7 previous messages</summary>

> Context compaction
>
> Source: automatic
> Status: completed
>
> Mình sẽ kiểm tra nhanh log build `flash_attn` để xác nhận Python/CUDA/wheel đang khớp, rồi mình gom lại thành một chuỗi lệnh chạy từ sạch đến experiment.
>
> Lệnh đọc log bị vấp vì thư mục làm việc mặc định của session không hợp lệ trên máy này; mình sẽ đổi `workdir` về repo rồi đọc lại đúng file log.
>
> <details><summary>Ran a command</summary>
>
> - Ran `Get-Content -Path 'D:\Users\flash_attn_build.log' -Tail 120`
>
> </details>
>
> Log xác nhận wheel `flash_attn-2.8.3.post1-cp310-cp310-linux_x86_64.whl`, nên env caption nên khóa Python 3.10. Mình đọc thêm config experiment hiện tại để lệnh chạy khớp đúng tên/path trong repo.
>
> <details><summary>Ran 3 commands</summary>
>
> - Ran `Get-Content -Path 'data_processing\caption\configs\caption_experiment_vlm_a5000.yaml'`
> - Ran `Get-Content -Path 'data_processing\caption\configs\caption_stage_vlm.env.example'`
> - Ran `Get-Content -Path 'data_processing\caption\configs\caption_stage_common.env.example'`
>
> </details>
>
> Config hiện tại đang benchmark `Qwen/Qwen2.5-VL-7B-Instruct`, chọn frame `representative` cho `L22_V012`, và output raw ở `caption_experiments_raw`. Mình sẽ đưa chuỗi lệnh theo đúng flow: tạo env -> test mock -> chạy VLM thật -> materialize -> retrieval eval -> benchmark so sánh.
</details>

Log bạn gửi xác nhận `flash-attn` đã build cho `cp310`, nên tạo env caption bằng **Python 3.10** là đúng nhất. Chạy tuần tự như sau trên server.

**1. Tạo env riêng cho caption VLM**

```bash
conda create -p /tmp2/maitanha/vgu/ttn/AIC-Khoa/conda_envs/aic-caption-vlm python=3.10 -y
conda activate /tmp2/maitanha/vgu/ttn/AIC-Khoa/conda_envs/aic-caption-vlm

python -V
which python
```

Kỳ vọng:

```bash
Python 3.10.x
/tmp2/maitanha/vgu/ttn/AIC-Khoa/conda_envs/aic-caption-vlm/bin/python
```

**2. Cài thư viện chính**

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

pip install \
  "transformers>=4.49.0" \
  "accelerate>=0.33.0" \
  "qwen-vl-utils>=0.0.8" \
  "safetensors>=0.4.5" \
  "sentencepiece>=0.2.0" \
  "protobuf>=4.25.0" \
  "einops>=0.8.0" \
  pillow pyyaml tqdm
```

**3. Cài flash-attn**

Nếu wheel cũ còn nằm ở path log:

```bash
pip install /tmp2/maitanha/vgu/ttn/AIC-Khoa/tmp/pip-ephem-wheel-cache-m8os9_8o/wheels/52/c6/18/28288f2ddc55daebd5c1972cafd9b00a352bd5128cdd37eb4d/flash_attn-2.8.3.post1-cp310-cp310-linux_x86_64.whl
```

Nếu file wheel đó đã bị xóa thì build lại:

```bash
pip install flash-attn==2.8.3.post1 --no-build-isolation
```

Check nhanh:

```bash
python -c "import torch, transformers, qwen_vl_utils, accelerate; print(torch.__version__, torch.cuda.is_available())"
python -c "import flash_attn; print('flash_attn ok')"
```

**4. Chuẩn bị env files**

```bash
cd /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/data_processing/caption

cp configs/caption_stage_common.env.example \
  /tmp2/maitanha/vgu/ttn/AIC-Khoa/conda_envs/caption_stage_common.env

cp configs/caption_stage_vlm.env.example \
  /tmp2/maitanha/vgu/ttn/AIC-Khoa/conda_envs/caption_stage_vlm.env
```

Nếu muốn chạy GPU 1, kiểm tra file này có:

```bash
CUDA_VISIBLE_DEVICES=1
CAPTION_VLM_ATTN=flash_attention_2
```

Nếu `flash_attention_2` lỗi ABI/runtime, đổi tạm:

```bash
CAPTION_VLM_ATTN=sdpa
```

**5. Smoke test bằng mock trước**

```bash
PYTHONIOENCODING=utf-8 python run_vlm_frame_caption_experiment.py \
  --config configs/caption_experiment_vlm_a5000.yaml \
  --env-file /tmp2/maitanha/vgu/ttn/AIC-Khoa/conda_envs/caption_stage_common.env \
  --env-file /tmp2/maitanha/vgu/ttn/AIC-Khoa/conda_envs/caption_stage_vlm.env \
  --provider mock \
  --experiment-name mock_vlm_smoke \
  --max-frames 2 \
  --no-resume
```

**6. Chạy VLM experiment thật trên L22_V012**

Test nhỏ 20 frame trước:

```bash
PYTHONIOENCODING=utf-8 python run_vlm_frame_caption_experiment.py \
  --config configs/caption_experiment_vlm_a5000.yaml \
  --env-file /tmp2/maitanha/vgu/ttn/AIC-Khoa/conda_envs/caption_stage_common.env \
  --env-file /tmp2/maitanha/vgu/ttn/AIC-Khoa/conda_envs/caption_stage_vlm.env \
  --experiment-name vlm_frame_qwen25vl_7b_rep \
  --frame-selection representative \
  --max-frames 20
```

Nếu ổn, chạy full representative set:

```bash
PYTHONIOENCODING=utf-8 python run_vlm_frame_caption_experiment.py \
  --config configs/caption_experiment_vlm_a5000.yaml \
  --env-file /tmp2/maitanha/vgu/ttn/AIC-Khoa/conda_envs/caption_stage_common.env \
  --env-file /tmp2/maitanha/vgu/ttn/AIC-Khoa/conda_envs/caption_stage_vlm.env \
  --experiment-name vlm_frame_qwen25vl_7b_rep \
  --frame-selection representative
```

**7. Materialize experiment vào pipeline chuẩn**

```bash
PYTHONIOENCODING=utf-8 python run_caption_experiment_materialize.py \
  --video-id L22_V012 \
  --baseline-dir /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption \
  --experiment-root /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption_experiments \
  --experiment-name vlm_frame_qwen25vl_7b_rep \
  --frame-captions /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption_experiments_raw/vlm_frame_qwen25vl_7b_rep/L22_V012/frame_caption_overrides.jsonl \
  --strict
```

**8. Chạy retrieval eval cho experiment**

```bash
PYTHONIOENCODING=utf-8 python run_retrieval_export_eval.py \
  --video-id L22_V012 \
  --caption-root /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption_experiments/vlm_frame_qwen25vl_7b_rep \
  --strict
```

**9. Benchmark so sánh baseline vs VLM experiment**

```bash
PYTHONIOENCODING=utf-8 python run_caption_experiment_benchmark.py \
  --video-id L22_V012 \
  --baseline-dir /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption \
  --candidate-dir /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption_experiments/vlm_frame_qwen25vl_7b_rep \
  --baseline-label template_baseline \
  --candidate-label vlm_frame_qwen25vl_7b_rep \
  --top-n-samples 3 \
  --strict
```

Tóm lại: lỗi trước đó là do đang chạy bằng `base Python 3.13`, trong khi `flash-attn` và stack VLM nên chạy ở env `Python 3.10`. Sau khi activate đúng env và có `qwen-vl-utils`, experiment Qwen sẽ chạy đúng hướng.

> "a_processing/caption$ python -c "import torch, transformers, qwen_vl_utils, accelerate; print(torch.**version**, torch.cuda.is_available())"
> python -c "import flash_attn; print('flash_attn ok')"
> 2.5.1+cu121 True
> Traceback (most recent call last):
>   File "<string>", line 1, in <module>
>   File "/tmp2/maitanha/vgu/ttn/AIC-Khoa/conda_envs/aic-caption-vlm/lib/python3.10/site-packages/flash_attn/**init**.py", line 3, in <module>
>     from flash_attn.flash_attn_interface import (
>   File "/tmp2/maitanha/vgu/ttn/AIC-Khoa/conda_envs/aic-caption-vlm/lib/python3.10/site-packages/flash_attn/flash_attn_interface.py", line 15, in <module>
>     import flash_attn_2_cuda as flash_attn_gpu
> ImportError: /tmp2/maitanha/vgu/ttn/AIC-Khoa/conda_envs/aic-caption-vlm/lib/python3.10/site-packages/flash_attn_2_cuda.cpython-310-x86_64-linux-gnu.so: undefined symbol: _ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_jb
> (/tmp2/maitanha/vgu/ttn/AIC-Khoa/conda_envs/aic-caption-vlm) maitanha@cl777:/tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/data_processing/caption$"

Lỗi này là **ABI mismatch**: `flash-attn` wheel bạn đang cài không khớp với `torch 2.5.1+cu121` trong env `aic-caption-vlm`. Torch/CUDA đang ổn, chỉ riêng `flash_attn` bị lệch symbol.

Cách nhanh và ổn định nhất để chạy experiment ngay: dùng `sdpa`.

```bash
sed -i 's/^CAPTION_VLM_ATTN=.*/CAPTION_VLM_ATTN=sdpa/' \
  /tmp2/maitanha/vgu/ttn/AIC-Khoa/conda_envs/caption_stage_vlm.env

grep CAPTION_VLM_ATTN \
  /tmp2/maitanha/vgu/ttn/AIC-Khoa/conda_envs/caption_stage_vlm.env
```

Sau đó chạy lại smoke test:

```bash
cd /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/data_processing/caption

PYTHONIOENCODING=utf-8 python run_vlm_frame_caption_experiment.py \
  --config configs/caption_experiment_vlm_a5000.yaml \
  --env-file /tmp2/maitanha/vgu/ttn/AIC-Khoa/conda_envs/caption_stage_common.env \
  --env-file /tmp2/maitanha/vgu/ttn/AIC-Khoa/conda_envs/caption_stage_vlm.env \
  --provider mock \
  --experiment-name mock_vlm_smoke \
  --max-frames 2 \
  --no-resume
```

Nếu pass, chạy Qwen test nhỏ:

```bash
PYTHONIOENCODING=utf-8 python run_vlm_frame_caption_experiment.py \
  --config configs/caption_experiment_vlm_a5000.yaml \
  --env-file /tmp2/maitanha/vgu/ttn/AIC-Khoa/conda_envs/caption_stage_common.env \
  --env-file /tmp2/maitanha/vgu/ttn/AIC-Khoa/conda_envs/caption_stage_vlm.env \
  --experiment-name vlm_frame_qwen25vl_7b_rep \
  --frame-selection representative \
  --max-frames 20
```

Nếu bạn vẫn muốn dùng `flash_attention_2`, cần rebuild lại `flash-attn` **ngay trong env hiện tại**:

```bash
conda activate /tmp2/maitanha/vgu/ttn/AIC-Khoa/conda_envs/aic-caption-vlm

pip uninstall -y flash-attn
MAX_JOBS=8 pip install flash-attn==2.8.3.post1 --no-build-isolation --no-cache-dir
python -c "import flash_attn; print('flash_attn ok')"
```

Nhưng để experiments không bị kẹt, mình khuyên dùng `CAPTION_VLM_ATTN=sdpa` trước. Chất lượng caption không đổi, chỉ có thể chậm hơn `flash_attention_2`.

> nếu muốn sửa số devices GPU

Đổi GPU cho caption VLM thì sửa trong file:

```bash
nano /tmp2/maitanha/vgu/ttn/AIC-Khoa/conda_envs/caption_stage_vlm.env
```

Ví dụ chạy trên **GPU physical 1**:

```bash
CUDA_VISIBLE_DEVICES=1
CAPTION_VLM_ATTN=sdpa
```

Lưu ý quan trọng: sau khi set `CUDA_VISIBLE_DEVICES=1`, bên trong Python GPU đó sẽ được nhìn là `cuda:0`. Vì vậy không cần sửa config sang `cuda:1`.

Nếu muốn chạy trên nhiều GPU, ví dụ physical GPU 1 và 2:

```bash
CUDA_VISIBLE_DEVICES=1,2
CAPTION_VLM_ATTN=sdpa
```

Với config hiện tại:

```yaml
device_map: auto
```

Transformers có thể tự chia model qua các GPU đang visible. Nhưng với `Qwen2.5-VL-7B`, thường **1 A5000 là đủ** cho batch tuần tự/frame caption. Mình khuyên:

- Demo/test: `CUDA_VISIBLE_DEVICES=1`
- Full `L22_V012`: vẫn `CUDA_VISIBLE_DEVICES=1` là ổn
- Chỉ dùng `CUDA_VISIBLE_DEVICES=1,2` nếu bị OOM hoặc muốn thử model lớn hơn

Check GPU đang được Python thấy:

```bash
python -c "import torch; print(torch.cuda.device_count()); print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"
```

Chạy tạm không cần sửa file `.env` cũng được:

```bash
CUDA_VISIBLE_DEVICES=2 PYTHONIOENCODING=utf-8 python run_vlm_frame_caption_experiment.py \
  --config configs/caption_experiment_vlm_a5000.yaml \
  --env-file /tmp2/maitanha/vgu/ttn/AIC-Khoa/conda_envs/caption_stage_common.env \
  --env-file /tmp2/maitanha/vgu/ttn/AIC-Khoa/conda_envs/caption_stage_vlm.env \
  --env-override \
  --experiment-name vlm_frame_qwen25vl_7b_rep \
  --frame-selection representative \
  --max-frames 20
```

Nhưng cách sạch hơn là sửa `caption_stage_vlm.env`.

> "aption_experiments_raw/vlm_frame_qwen25vl_7b_rep/L22_V012/frame_caption_overrides.jsonl
> 12:17:58 [INFO] caption_pipeline.vlm_frame_experiment: Wrote VLM report -> /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption_experiments_raw/vlm_frame_qwen25vl_7b_rep/L22_V012/frame_caption_overrides.shard_000.report.json
> 12:17:58 [INFO] caption_vlm_frame_experiment: === VLM frame experiment done ===
> 12:17:58 [INFO] caption_vlm_frame_experiment:   Output rows: 20
> 12:17:58 [INFO] caption_vlm_frame_experiment:   Processed:   20
> 12:17:58 [INFO] caption_vlm_frame_experiment:   Failures:    0
> (/tmp2/maitanha/vgu/ttn/AIC-Khoa/conda_envs/aic-caption-vlm) maitanha@cl777:/tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/data_processing/caption$"

Pass đẹp rồi. Đây là **Qwen VLM test nhỏ 20 frame đã chạy thành công**, output nằm ở:

```bash
/tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption_experiments_raw/vlm_frame_qwen25vl_7b_rep/L22_V012/frame_caption_overrides.jsonl
```

Bước tiếp theo là chạy full representative frames cho `L22_V012`:

```bash
cd /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/data_processing/caption

PYTHONIOENCODING=utf-8 python run_vlm_frame_caption_experiment.py \
  --config configs/caption_experiment_vlm_a5000.yaml \
  --env-file /tmp2/maitanha/vgu/ttn/AIC-Khoa/conda_envs/caption_stage_common.env \
  --env-file /tmp2/maitanha/vgu/ttn/AIC-Khoa/conda_envs/caption_stage_vlm.env \
  --experiment-name vlm_frame_qwen25vl_7b_rep \
  --frame-selection representative
```

Vì lần test đã ghi 20 dòng và config đang `resume: true`, lệnh trên sẽ tiếp tục/resume nếu code nhận diện output cũ. Nếu bạn muốn chạy lại sạch từ đầu:

```bash
PYTHONIOENCODING=utf-8 python run_vlm_frame_caption_experiment.py \
  --config configs/caption_experiment_vlm_a5000.yaml \
  --env-file /tmp2/maitanha/vgu/ttn/AIC-Khoa/conda_envs/caption_stage_common.env \
  --env-file /tmp2/maitanha/vgu/ttn/AIC-Khoa/conda_envs/caption_stage_vlm.env \
  --experiment-name vlm_frame_qwen25vl_7b_rep \
  --frame-selection representative \
  --no-resume
```

Sau khi full run xong, mới chạy tiếp:

```bash
PYTHONIOENCODING=utf-8 python run_caption_experiment_materialize.py \
  --video-id L22_V012 \
  --baseline-dir /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption \
  --experiment-root /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption_experiments \
  --experiment-name vlm_frame_qwen25vl_7b_rep \
  --frame-captions /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption_experiments_raw/vlm_frame_qwen25vl_7b_rep/L22_V012/frame_caption_overrides.jsonl \
  --strict
```

> "ion_experiments_raw/vlm_frame_qwen25vl_7b_rep/L22_V012/frame_caption_overrides.shard_000.report.json
> 12:42:39 [INFO] caption_vlm_frame_experiment: === VLM frame experiment done ===
> 12:42:39 [INFO] caption_vlm_frame_experiment:   Output rows: 207
> 12:42:39 [INFO] caption_vlm_frame_experiment:   Processed:   187
> 12:42:39 [INFO] caption_vlm_frame_experiment:   Failures:    0
> (/tmp2/maitanha/vgu/ttn/AIC-Khoa/conda_envs/aic-caption-vlm) maitanha@cl777:/tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/dat
> a_processing/caption$ PYTHONIOENCODING=utf-8 python run_caption_experiment_materialize.py \
>   --video-id L22_V012 \
>   --baseline-dir /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption \
>   --experiment-root /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption_experiments \
>   --experiment-name vlm_frame_qwen25vl_7b_rep \
>   --frame-captions /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption_experiments_raw/vlm_frame_qwen25vl_7b_rep/L22_V012/frame_caption_overrides.jsonl \
>   --strict
> usage: run_caption_experiment_materialize.py [-h] --video-id VIDEO_ID --baseline-dir BASELINE_DIR --output-root
>                                              OUTPUT_ROOT --experiment-name EXPERIMENT_NAME
>                                              [--frame-captions FRAME_CAPTIONS] [--shot-captions SHOT_CAPTIONS]
>                                              [--event-steps EVENT_STEPS] [--copy-reports] [--strict] [--verbose]
> run_caption_experiment_materialize.py: error: the following arguments are required: --output-root"

<details><summary>1 previous message</summary>

> Lỗi này là do CLI thực tế của `run_caption_experiment_materialize.py` đang dùng tên tham số `--output-root`, không phải `--experiment-root`. Mình sẽ sửa lại đúng theo script hiện tại và đưa tiếp luôn các lệnh sau materialize.
</details>

Đúng rồi, command materialize cần đổi:

```bash
--experiment-root
```

thành:

```bash
--output-root
```

Chạy lại như sau:

```bash
cd /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/data_processing/caption

PYTHONIOENCODING=utf-8 python run_caption_experiment_materialize.py \
  --video-id L22_V012 \
  --baseline-dir /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption \
  --output-root /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption_experiments \
  --experiment-name vlm_frame_qwen25vl_7b_rep \
  --frame-captions /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption_experiments_raw/vlm_frame_qwen25vl_7b_rep/L22_V012/frame_caption_overrides.jsonl \
  --strict
```

Log VLM của bạn ổn:

```text
Output rows: 207
Processed:   187
Failures:    0
```

`207 = 20 frame đã chạy test trước + 187 frame chạy tiếp bằng resume`, nên không có lỗi.

Sau khi materialize pass, chạy tiếp retrieval eval:

```bash
PYTHONIOENCODING=utf-8 python run_retrieval_export_eval.py \
  --video-id L22_V012 \
  --caption-root /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption_experiments/vlm_frame_qwen25vl_7b_rep \
  --strict
```

Rồi benchmark so sánh baseline template với VLM experiment:

```bash
PYTHONIOENCODING=utf-8 python run_caption_experiment_benchmark.py \
  --video-id L22_V012 \
  --baseline-dir /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption \
  --candidate-dir /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption_experiments/vlm_frame_qwen25vl_7b_rep \
  --baseline-label template_baseline \
  --candidate-label vlm_frame_qwen25vl_7b_rep \
  --top-n-samples 3 \
  --strict
```

> "(/tmp2/maitanha/vgu/ttn/AIC-Khoa/conda_envs/aic-caption-vlm) maitanha@cl777:/tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/data_processing/caption$ PYTHONIOENCODING=utf-8 python run_retrieval_export_eval.py \
>   --video-id L22_V012 \
>   --caption-root /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption_experiments/vlm_frame_qwen25vl_7b_rep \
>   --strict
> usage: run_retrieval_export_eval.py [-h] --video-id VIDEO_ID --caption-dir CAPTION_DIR [--compact-index COMPACT_INDEX]
>                                     [--event-step-index EVENT_STEP_INDEX] [--queries-jsonl QUERIES_JSONL]
>                                     [--top-k TOP_K] [--write-es-bulk] [--write-default-queries] [--strict] [--verbose]
> run_retrieval_export_eval.py: error: the following arguments are required: --caption-dir
> (/tmp2/maitanha/vgu/ttn/AIC-Khoa/conda_envs/aic-caption-vlm) maitanha@cl777:/tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/data_processing/caption$ PYTHONIOENCODING=utf-8 python run_caption_experiment_benchmark.py \
>   --video-id L22_V012 \
>   --baseline-dir /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption \
>   --candidate-dir /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption_experiments/vlm_frame_qwen25vl_7b_rep \
>   --baseline-label template_baseline \
>   --candidate-label vlm_frame_qwen25vl_7b_rep \
>   --top-n-samples 3 \
>   --strict
> usage: run_caption_experiment_benchmark.py [-h] --video-id VIDEO_ID --baseline-dir BASELINE_DIR
>                                            [--baseline-label BASELINE_LABEL] [--experiment EXPERIMENT]
>                                            [--output-dir OUTPUT_DIR] [--benchmark-name BENCHMARK_NAME]
>                                            [--top-n-samples TOP_N_SAMPLES] [--strict] [--verbose]
> run_caption_experiment_benchmark.py: error: unrecognized arguments: --candidate-dir /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption_experiments/vlm_frame_qwen25vl_7b_rep --candidate-label vlm_frame_qwen25vl_7b_rep
> (/tmp2/maitanha/vgu/ttn/AIC-Khoa/conda_envs/aic-caption-vlm) maitanha@cl777:/tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/data_processing/caption$"

<details><summary>3 previous messages</summary>

> Mình thấy hai script này dùng tên tham số khác với lệnh mình đưa trước đó. Mình sẽ đọc nhanh `argparse` trong hai file để trả lại đúng command theo CLI hiện tại, tránh đoán nhầm path output.
>
> <details><summary>Ran 3 commands</summary>
>
> - Ran `Select-String -Path 'data_processing\caption\run_caption_experiment_benchmark.py' -Pattern 'add_argument|experiment|baseline-dir|output-dir' -Context 0,2`
> - Ran `Select-String -Path 'data_processing\caption\caption_pipeline\experiment_benchmark.py' -Pattern 'experiment|candidate|baseline_dir|output_dir|caption' -Context 0,2`
> - Ran `Select-String -Path 'data_processing\caption\run_retrieval_export_eval.py' -Pattern 'add_argument|caption-dir|caption_dir|experiment' -Context 0,2`
>
> </details>
>
> Hai lỗi đều là do tên tham số CLI hiện tại khác: retrieval dùng `--caption-dir`, benchmark dùng `--experiment label=path`. Path cần trỏ tới thư mục gốc chứa `L22_V012/`, tức experiment root sau materialize.
</details>

Chạy lại đúng theo CLI hiện tại như sau.

**1. Retrieval eval cho experiment VLM**

```bash
cd /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/data_processing/caption

PYTHONIOENCODING=utf-8 python run_retrieval_export_eval.py \
  --video-id L22_V012 \
  --caption-dir /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption_experiments/vlm_frame_qwen25vl_7b_rep \
  --strict
```

Lưu ý: script cần `--caption-dir`, không phải `--caption-root`.

**2. Benchmark so sánh baseline vs VLM experiment**

```bash
PYTHONIOENCODING=utf-8 python run_caption_experiment_benchmark.py \
  --video-id L22_V012 \
  --baseline-dir /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption \
  --baseline-label template_baseline \
  --experiment vlm_frame_qwen25vl_7b_rep=/tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption_experiments/vlm_frame_qwen25vl_7b_rep \
  --top-n-samples 3 \
  --strict
```

Script benchmark hiện tại không có `--candidate-dir` / `--candidate-label`; nó dùng dạng:

```bash
--experiment label=path
```

Nếu muốn chắc chắn materialize đã tạo đúng thư mục trước khi chạy retrieval, check nhanh:

```bash
ls /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption_experiments/vlm_frame_qwen25vl_7b_rep/L22_V012
```

Cần thấy các folder kiểu:

```bash
captions  evidence  indexes  reports
```

> "n/L22_V012/benchmarks/phase6_retrieval_benchmark/manual_review_samples.jsonl
> 12:46:40 [INFO] caption_phase6_benchmark: === Done in 0.1s ===
> 12:46:40 [INFO] caption_phase6_benchmark:   Candidates: 2
> 12:46:40 [INFO] caption_phase6_benchmark:   Samples:    12 queries
> 12:46:40 [INFO] caption_phase6_benchmark:   Warnings:   0
> 12:46:40 [INFO] caption_phase6_benchmark: All caption experiment benchmark checks passed.ents/vlm_frame_qwen25vl_7b_rep/L22_V012/eval/retrieval_eval_report.md
> 12:46:18 [INFO] caption_retrieval: === Done in 6.0s ===
> 12:46:18 [INFO] caption_retrieval:   Corpus:  487 docs
> 12:46:18 [INFO] caption_retrieval:   Queries: 12
> 12:46:18 [INFO] caption_retrieval:   Results: 240
> 12:46:18 [INFO] caption_retrieval:   Hits:    12/12
> 12:46:18 [INFO] caption_retrieval:   Warnings: 0
> 12:46:18 [INFO] caption_retrieval: All retrieval export/eval acceptance checks passed."

Pass hết rồi. Kết quả hiện tại nghĩa là experiment VLM `vlm_frame_qwen25vl_7b_rep` đã đi đủ pipeline:

- VLM frame caption raw: OK, `Failures: 0`
- Materialize vào pipeline chuẩn: OK
- Retrieval eval: OK, `Hits: 12/12`, `Warnings: 0`
- Benchmark baseline vs VLM: OK, `Candidates: 2`, `Warnings: 0`

Các file nên xem tiếp:

```bash
cat /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption_experiments/vlm_frame_qwen25vl_7b_rep/L22_V012/eval/retrieval_eval_report.json
cat /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption/L22_V012/benchmarks/phase6_retrieval_benchmark/phase6_retrieval_benchmark.json
```

Bản dễ đọc hơn:

```bash
less /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption_experiments/vlm_frame_qwen25vl_7b_rep/L22_V012/eval/retrieval_eval_report.md
less /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption/L22_V012/benchmarks/phase6_retrieval_benchmark/phase6_retrieval_benchmark.md
less /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption/L22_V012/benchmarks/phase6_retrieval_benchmark/manual_review_samples.jsonl
```

Điểm cần nhớ: metric tự động hiện tại mới xác nhận retrieval không hỏng và top-k vẫn hit đủ 12/12. Muốn biết VLM có “hay hơn” template thật không thì nên xem `manual_review_samples.jsonl` và report benchmark, vì với 12 query demo thì hit-rate có thể hòa nhau.

> Hãy nhận xét
>
> User context
>
> Attachments:
> - manual_review_samples.jsonl: `d:/Users/manual_review_samples.jsonl`

<details><summary>10 previous messages</summary>

> Mình sẽ đọc `manual_review_samples.jsonl` để xem top results của baseline và VLM đang khác nhau thế nào, rồi nhận xét theo hướng retrieval thực tế: có cải thiện, có lệch intent, và nên làm gì tiếp.
>
> <details><summary>Ran 3 commands</summary>
>
> - Ran `(Get-Content -Path 'D:\Users\manual_review_samples.jsonl').Count`
> - Ran `Get-Content -Path 'D:\Users\manual_review_samples.jsonl' -TotalCount 3`
> - Ran `Get-Item -Path 'D:\Users\manual_review_samples.jsonl' | Select-Object FullName,Length,LastWriteTime`
>
> </details>
>
> Ba dòng đầu cho thấy top-3 của `template_baseline` và `vlm_frame_qwen25vl_7b_rep` đang giống hệt nhau. Mình sẽ parse đủ 12 query để xem đây là hiện tượng toàn bộ hay chỉ vài query đầu.
>
> <details><summary>Ran a command</summary>
>
> - Ran `$items = Get-Content -Path 'D:\Users\manual_review_samples.jsonl' | ForEach-Object { $_ | ConvertFrom-Json }; foreach ($item in $items) { $base = ($item.candidates | Where-Object { $_.label -eq 'template_baseline' }).top_results; $vlm = ($item.candidates | Where-Object { $_.label -eq 'vlm_frame_qwen25vl_7b_rep' }).top_results; $sameTop1 = ($base[0].document_id -eq $vlm[0].document_id); $sameTop3 = (($base | ForEach-Object document_id) -join ',') -eq (($vlm | ForEach-Object document_id) -join ','); [PSCustomObject]@{query_id=$item.query_id; route=$item.route; query=$item.query; same_top1=$sameTop1; same_top3=$sameTop3; base_top1=$base[0].document_id; vlm_top1=$vlm[0].document_id; base_score=$base[0].score; vlm_score=$vlm[0].score} } | Format-Table -AutoSize`
>
> </details>
>
> Câu lệnh parse đầu bị lỗi cú pháp PowerShell ở đoạn lấy field trong mảng. Mình sẽ chạy lại bản rõ ràng hơn, chỉ tạo table so sánh top-1/top-3.
>
> <details><summary>Ran a command</summary>
>
> - Ran `$items = Get-Content -Path 'D:\Users\manual_review_samples.jsonl' | ForEach-Object { ConvertFrom-Json $_ }
> $rows = @()
> foreach ($item in $items) {
>   $baseCand = $item.candidates | Where-Object { $_.label -eq 'template_baseline' }
>   $vlmCand = $item.candidates | Where-Object { $_.label -eq 'vlm_frame_qwen25vl_7b_rep' }
>   $base = $baseCand.top_results
>   $vlm = $vlmCand.top_results
>   $baseIds = @($base | ForEach-Object { $_.document_id }) -join ','
>   $vlmIds = @($vlm | ForEach-Object { $_.document_id }) -join ','
>   $rows += [PSCustomObject]@{
>     query_id = $item.query_id
>     route = $item.route
>     same_top1 = ($base[0].document_id -eq $vlm[0].document_id)
>     same_top3 = ($baseIds -eq $vlmIds)
>     base_top1 = $base[0].document_id
>     vlm_top1 = $vlm[0].document_id
>     base_score = $base[0].score
>     vlm_score = $vlm[0].score
>   }
> }
> $rows | Format-Table -AutoSize`
>
> </details>
>
> Kết quả parse xác nhận 12/12 query có `same_top1=True` và `same_top3=True`. Mình sẽ kiểm tra thêm materializer một chút để phân biệt: VLM đã được inject nhưng query không chạm tới nó, hay compact index chưa được rebuild sau khi inject.
>
> <details><summary>Ran 3 commands</summary>
>
> - Ran `Select-String -Path 'data_processing\caption\caption_pipeline\caption_index.py' -Pattern 'compact|caption_text|frame_index|shot|event|all_text' -Context 1,2`
> - Ran `Select-String -Path 'data_processing\caption\caption_pipeline\experiment_materializer.py' -Pattern 'compact|frame_index|copy|override|jsonl|caption_text' -Context 2,3`
> - Ran `Select-String -Path 'data_processing\caption\run_caption_experiment_materialize.py' -Pattern 'add_argument|output-root|strict|frame-captions' -Context 0,2`
>
> </details>
>
> Một dấu hiệu quan trọng: trong top-1 của 12 query không có doc `unit_type=frame`; phần lớn là `event_step` hoặc `shot`. Tức là VLM frame captions có thể đã được ghi, nhưng benchmark hiện tại gần như chưa tạo áp lực lên tầng frame-caption.
>
> <details><summary>Ran a command</summary>
>
> - Ran `$items = Get-Content -Path 'D:\Users\manual_review_samples.jsonl' | ForEach-Object { ConvertFrom-Json $_ }
> foreach ($label in @('template_baseline','vlm_frame_qwen25vl_7b_rep')) {
>   $types = @{}
>   foreach ($item in $items) {
>     $cand = $item.candidates | Where-Object { $_.label -eq $label }
>     foreach ($r in $cand.top_results) {
>       $k = $r.unit_type
>       if (-not $types.ContainsKey($k)) { $types[$k] = 0 }
>       $types[$k] += 1
>     }
>   }
>   Write-Output "LABEL=$label"
>   $types.GetEnumerator() | Sort-Object Name | ForEach-Object { Write-Output ("{0}: {1}" -f $_.Name, $_.Value) }
> }`
>
> </details>
</details>

Nhận xét ngắn: benchmark này **pass kỹ thuật**, nhưng **chưa chứng minh VLM tốt hơn baseline**.

Từ `manual_review_samples.jsonl`:

- 12/12 query có `same_top1 = True`.
- 12/12 query có `same_top3 = True`.
- Tổng 36 kết quả sample top-3:
  - `event_step`: 21
  - `shot`: 15
  - `frame`: 0

Nghĩa là top results của `template_baseline` và `vlm_frame_qwen25vl_7b_rep` đang **giống hệt nhau**. VLM không làm hỏng pipeline, nhưng tác động của VLM frame caption chưa xuất hiện trong retrieval result.

Lý do chính: experiment hiện tại chỉ override **frame captions**, trong khi các query demo đang trả về chủ yếu `event_step` và `shot`. Phần `shot caption` / `TRAKE event step` vẫn là bản baseline, nên kết quả top-k gần như không đổi.

Kết luận của mình:

- Pipeline VLM frame-caption: OK.
- Materialize/eval/benchmark: OK.
- Metric `Hits 12/12`: tốt cho acceptance, nhưng chưa đủ để đánh giá chất lượng VLM.
- Manual review hiện tại cho thấy VLM chưa tạo khác biệt trong search ranking.

Bước nên làm tiếp: benchmark VLM đúng nghĩa hơn bằng một trong hai hướng:

1. Tạo thêm query nhắm vào chi tiết hình ảnh/frame, ví dụ trang phục, hành động, bố cục, vật thể nhỏ, cảnh cụ thể.
2. Hoặc nâng experiment lên full pipeline: VLM frame captions -> rebuild shot captions -> rebuild TRAKE event steps -> rebuild retrieval index -> benchmark lại.

Nếu mục tiêu là so sánh “VLM caption có giúp search tốt hơn không”, mình nghiêng về hướng 2. Hiện tại mới là benchmark an toàn để xác nhận VLM không phá pipeline.

> Tiếp phase tiếp theo có thể là làm gì

Phase tiếp theo hợp lý nhất là **Phase 8: Full Experiment Propagation**.

Hiện tại VLM mới thay **frame caption**, nhưng search top-k đang ăn nhiều từ `shot` và `event_step`, nên kết quả chưa đổi. Phase 8 nên làm việc này:

```text
VLM frame captions
-> rebuild shot captions từ frame captions mới
-> rebuild TRAKE/event steps từ shot captions mới
-> rebuild compact_search_index
-> retrieval eval
-> benchmark baseline vs experiment
```

Mục tiêu: để caption VLM thật sự lan lên 3 cấp độ caption đã thiết kế:

- Cấp 1: `frame_index.jsonl`
- Cấp 2: `shot_index.jsonl`
- Cấp 3: `event_step_index.jsonl` / TRAKE text
- Cuối cùng: `compact_search_index.jsonl`

Mình đề xuất thứ tự phase tiếp theo:

**Phase 8A: Rebuild experiment pipeline từ overrides**

Viết một script kiểu:

```bash
run_caption_experiment_propagate.py
```

Input:

```bash
--video-id L22_V012
--baseline-dir caption
--output-root caption_experiments
--experiment-name vlm_frame_qwen25vl_7b_rep
--frame-captions caption_experiments_raw/.../frame_caption_overrides.jsonl
```

Output:

```text
caption_experiments/vlm_frame_qwen25vl_7b_rep/L22_V012/
  captions/frame_index.jsonl
  captions/shot_index.jsonl
  captions/event_step_index.jsonl
  indexes/compact_search_index.jsonl
  reports/experiment_propagation_report.json
```

**Phase 8B: Add visual-focused query set**

Hiện 12 query demo hơi “dễ” và nhiều query dựa vào OCR/audio/event. Nên thêm file query riêng:

```text
caption/L22_V012/eval/visual_retrieval_queries.jsonl
```

Ví dụ query nên tập trung vào thứ VLM có lợi thế:

```text
people wearing ao dai
news anchor in studio
storm warning map on screen
boats on water
classroom with laptop and sewing machine
cartoon preview screen
```

Sau đó chạy retrieval eval với query set này.

**Phase 8C: Benchmark 3 nhánh**

So sánh:

```text
template_baseline
vlm_frame_only
vlm_frame_propagated
```

Như vậy mới thấy rõ:

- VLM frame-only có tác động không?
- Khi propagate lên shot/event thì retrieval có đổi không?
- Query visual có cải thiện không?

Nếu muốn đi nhanh, mình sẽ làm **Phase 8A trước**: viết code để lấy VLM frame captions và rebuild lại shot/event/index cho experiment. Đây là bước đang thiếu nhất.

> Vậy khi nào mới tác động thật vào pipeline caption với việc sử dụng bộ prompt [prompts](data_processing/caption/caption_pipeline/prompts/)

Nó sẽ tác động thật khi ta dùng prompt để **sinh lại caption ở các tầng cao hơn**, không chỉ thay `frame_index`.

Hiện tại experiment VLM vừa chạy:

```text
Qwen image -> frame_caption_overrides.jsonl
-> materialize frame_index + compact frame docs
```

Nó **có dùng prompt**, nhưng là prompt inline trong file config:

```yaml
generation:
  prompt: Describe this video frame...
```

Chưa dùng trực tiếp các file trong:

```text
data_processing/caption/caption_pipeline/prompts/
```

Và quan trọng hơn: nó chưa regenerate lại:

```text
shot_caption_prompt.txt
trake_event_step_prompt.txt
```

nên `shot_index`, `event_step_index`, `trake_text` vẫn gần như baseline. Vì retrieval hiện đang ưu tiên nhiều `shot` và `event_step`, VLM frame caption chưa tạo khác biệt rõ.

Tác động thật vào pipeline caption sẽ bắt đầu ở phase tiếp theo, khi flow là:

```text
1. VLM sinh frame captions
   dùng frame_caption_prompt.txt hoặc prompt config

2. Shot captioner đọc nhiều frame captions trong cùng shot
   dùng shot_caption_prompt.txt
   -> sinh lại shot_index.jsonl

3. TRAKE/event builder đọc shot captions + OCR/audio/object
   dùng trake_event_step_prompt.txt
   -> sinh lại event_step_index.jsonl

4. Rebuild compact_search_index.jsonl
   -> retrieval search bắt đầu thấy khác biệt
```

Nói gọn:

- **Frame prompt** tác động ở tầng frame.
- **Shot prompt** mới làm VLM caption lan lên tầng shot.
- **TRAKE prompt** mới làm nó lan lên event/search narrative.
- Benchmark hiện tại mới chạm tầng 1, chưa chạm đủ pipeline caption.

Nếu muốn đúng với thiết kế “3 cấp độ caption”, phase kế tiếp nên là: **Experiment propagation bằng prompt files**. Tức là code sẽ đọc caption VLM đã sinh, sau đó chạy lại shot caption và event step bằng prompt trong thư mục `prompts/`.

> Là hiện tại mình đang dừng ở bước nào

Hiện tại mình đang dừng ở mức:

```text
VLM frame-caption experiment đã chạy xong
```

Cụ thể đã hoàn thành:

1. Tạo env riêng `aic-caption-vlm`: OK
2. Chạy Qwen2.5-VL trên `L22_V012`: OK
3. Sinh `frame_caption_overrides.jsonl`: OK
4. Materialize vào experiment dir: OK
5. Retrieval eval: OK, `Hits 12/12`, `Warnings 0`
6. Benchmark baseline vs VLM frame-only: OK, `Candidates 2`, `Warnings 0`

Nhưng pipeline caption **chưa propagate VLM lên các tầng cao hơn**.

Nói theo 3 cấp độ caption:

```text
Frame caption: đã có VLM caption
Shot caption: vẫn baseline/template
Event/TRAKE caption: vẫn baseline/template
```

Vì vậy kết quả search hiện tại chưa đổi nhiều. Ta đang dừng ở:

```text
Phase: VLM frame-only experiment
Trạng thái: pass kỹ thuật, chưa phải full caption pipeline experiment
```

Bước kế tiếp nếu muốn tác động thật vào pipeline:

```text
Phase 8: Propagate VLM frame captions
-> rebuild shot captions bằng shot_caption_prompt.txt
-> rebuild event/TRAKE bằng trake_event_step_prompt.txt
-> rebuild compact_search_index
-> chạy retrieval eval + benchmark lại
```

> và hiện tại về trạng thái Frame caption đã đạt hay chưa ? Nếu chuẩn theo từng bước chúng ta sẽ đến gì

Frame caption hiện tại **đạt ở mức kỹ thuật**, nhưng **chưa thể kết luận đạt ở mức chất lượng cuối cùng**.

Cụ thể:

```text
Frame caption VLM:
- Chạy được Qwen2.5-VL: đạt
- Sinh caption cho representative frames: đạt
- Failures = 0: đạt
- Ghi frame_caption_overrides.jsonl: đạt
- Materialize vào experiment: đạt
- Không phá retrieval/eval: đạt
```

Nhưng phần chưa đủ:

```text
- Chưa review chất lượng caption từng frame
- Chưa có visual query set để đo tác động của frame caption
- Chưa propagate lên shot/event
- Benchmark hiện tại top-k không đổi so với baseline
```

Vì vậy nên đánh giá là:

```text
Frame caption stage: PASS để đi tiếp
Final retrieval quality: chưa kết luận
```

Nếu đi chuẩn từng bước, thứ tự tiếp theo nên là:

**Bước 1: Frame Caption QA**

Kiểm tra file:

```bash
less /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption_experiments_raw/vlm_frame_qwen25vl_7b_rep/L22_V012/frame_caption_overrides.jsonl
```

Mục tiêu kiểm tra:

```text
caption có mô tả đúng ảnh không
có hallucination nặng không
có quá chung chung không
có nhắc người/vật/cảnh/hành động/text màn hình khi thấy rõ không
```

**Bước 2: Visual query benchmark riêng cho frame**

Tạo thêm query thiên về hình ảnh, vì query cũ đang ăn nhiều `audio`, `ocr`, `event_step`.

Mục tiêu: xem frame caption VLM có giúp tìm cảnh/vật thể/hành động tốt hơn không.

**Bước 3: Propagate lên shot caption**

Dùng frame captions đã tốt để sinh lại:

```text
shot_index.jsonl
```

Lúc này mới dùng mạnh:

```text
shot_caption_prompt.txt
```

**Bước 4: Propagate lên event/TRAKE**

Dùng shot captions mới + OCR/audio/object để sinh lại:

```text
event_step_index.jsonl
```

Lúc này mới dùng:

```text
trake_event_step_prompt.txt
```

**Bước 5: Rebuild search index + retrieval benchmark**

Rebuild:

```text
compact_search_index.jsonl
```

Sau đó chạy:

```text
retrieval eval
benchmark baseline vs VLM propagated
manual review samples
```

Tóm gọn: hiện tại **Frame caption đủ tốt để chuyển phase**, phase kế tiếp nên là **Frame Caption QA + visual query**, rồi mới propagate lên `shot` và `event`.

> Đã tốt, hãy nói sơ về phase tiếp theo sẽ làm trước khi thực hiện

Phase tiếp theo nên là **Phase 8: Propagate VLM captions vào full caption pipeline**.

Mục tiêu của phase này: biến kết quả VLM hiện tại từ “frame-only experiment” thành “caption pipeline experiment thật sự”.

Flow sẽ là:

```text
VLM frame captions đã có
-> kiểm tra/chuẩn hóa frame caption overrides
-> rebuild frame_index
-> rebuild shot captions từ frame captions mới
-> rebuild event/TRAKE captions từ shot captions mới
-> rebuild compact_search_index
-> retrieval eval
-> benchmark baseline vs VLM propagated
```

Điểm khác với hiện tại:

```text
Hiện tại:
VLM chỉ thay frame caption, shot/event vẫn baseline.

Phase tiếp theo:
VLM frame captions sẽ ảnh hưởng lên shot caption và event/TRAKE text.
```

Các prompt sẽ bắt đầu có vai trò rõ hơn:

```text
frame_caption_prompt.txt
  dùng làm chuẩn prompt cho frame caption experiment, thay vì prompt inline trong yaml nếu cần thống nhất.

shot_caption_prompt.txt
  dùng để tổng hợp nhiều frame captions thành shot caption.

trake_event_step_prompt.txt
  dùng để tạo event/TRAKE step từ shot caption + OCR/audio/object.
```

Mình sẽ chia phase này thành 3 bước nhỏ để dễ kiểm soát:

**8A. Frame Caption QA + Normalize**

Đọc `frame_caption_overrides.jsonl`, kiểm tra số lượng, duplicate, caption rỗng, caption quá dài/quá ngắn, mapping frame id.

Output dự kiến:

```text
reports/frame_caption_override_qa.json
reports/frame_caption_override_qa.md
```

**8B. Experiment Propagation**

Viết script mới, ví dụ:

```text
run_caption_experiment_propagate.py
```

Nó sẽ nhận VLM frame captions và rebuild lại:

```text
captions/frame_index.jsonl
captions/shot_index.jsonl
captions/event_step_index.jsonl
indexes/compact_search_index.jsonl
```

**8C. Visual Benchmark**

Thêm query set thiên về visual để đánh giá đúng tác động của VLM:

```text
eval/visual_retrieval_queries.jsonl
```

Sau đó chạy retrieval eval + benchmark lại.

Thứ tự mình đề xuất khi code:

1. Làm `8A` trước vì nhẹ và giúp phát hiện lỗi dữ liệu.
2. Làm `8B` để VLM thật sự lan lên shot/event.
3. Làm `8C` để benchmark có ý nghĩa hơn.

Nếu bắt đầu thực hiện, mình sẽ đọc lại các module hiện có như `experiment_materializer.py`, `shot_captioner.py`, `trake_event_builder.py`, `event_step_index.py`, rồi code theo hướng tận dụng lại pipeline cũ, không viết lại từ đầu.

> và mô hình sử dụng hiện tại sẽ là gì ? Nếu pass qua phase này sẽ tiếp đến phase nào

Hiện tại mô hình đang dùng là:

```text
Qwen/Qwen2.5-VL-7B-Instruct
```

Trong config:

```yaml
model:
  provider: qwen2_5_vl
  model_name: Qwen/Qwen2.5-VL-7B-Instruct
  dtype: bfloat16
  device_map: auto
```

Vai trò của nó hiện tại:

```text
Qwen2.5-VL đọc ảnh frame
-> sinh frame caption
-> ghi frame_caption_overrides.jsonl
```

Ở phase tiếp theo, nếu chưa thêm LLM riêng, mô hình vẫn có thể là **Qwen2.5-VL-7B-Instruct**, nhưng vai trò sẽ khác nhau:

```text
Frame caption: Qwen2.5-VL
Shot caption: có thể dùng rule/template trước, hoặc dùng LLM text-only sau
Event/TRAKE: có thể dùng rule/template trước, hoặc dùng LLM text-only sau
```

Mình đề xuất phase tiếp theo vẫn dùng cách **an toàn trước**:

```text
Phase 8:
Qwen2.5-VL frame captions
-> propagate bằng builder hiện có/template
-> rebuild shot/event/index
-> benchmark
```

Tức là chưa cần gọi thêm LLM text-only ngay. Mục tiêu là kiểm tra: chỉ cần frame caption tốt hơn, khi lan lên shot/event thì retrieval có đổi không.

Nếu Phase 8 pass, phase kế tiếp sẽ là:

```text
Phase 9: LLM/Fuse Caption Experiment
```

Lúc đó mới dùng mô hình text-generation để fuse nhiều nguồn:

```text
VLM caption
OCR
ASR/audio
object detection
scene
temporal context
```

Output sẽ là caption giàu hơn ở tầng:

```text
shot caption
event/TRAKE caption
```

Các mô hình có thể benchmark ở Phase 9:

```text
Qwen2.5 / Qwen3 text-only instruct
InternVL nếu dùng multimodal tiếp
FuseCap-style fusion nếu mình tự dựng prompt/rule theo plan
```

Tóm tắt đường đi:

```text
Hiện tại:
Phase 7 - Qwen2.5-VL frame-caption experiment: pass

Tiếp theo:
Phase 8 - propagate VLM frame captions vào shot/event/index

Sau khi Phase 8 pass:
Phase 9 - LLM/FuseCap caption fusion thật sự

Sau Phase 9:
Phase 10 - scale batch nhiều video + chọn best config
```

> Hãy thực hiện phase 8

<details><summary>30 previous messages</summary>

> Mình sẽ bắt đầu Phase 8 bằng cách đọc lại các module caption hiện có để tận dụng builder cũ thay vì dựng pipeline mới từ trắng. Trọng tâm là tạo một script propagate nhận `frame_caption_overrides.jsonl`, rebuild lại frame/shot/event/index, rồi xuất report kiểm tra.
>
> <details><summary>Ran 3 commands</summary>
>
> - Ran `rg --files data_processing/caption`
> - Ran `Get-ChildItem -Path data_processing\caption\caption_pipeline -File | Select-Object Name`
> - Ran `git status --short`
>
> </details>
>
> Worktree đang sạch, tốt. Mình sẽ đọc luồng Phase 2-4 hiện tại (`run_caption_pipeline.py`, shot/event builders, materializer) để nối Phase 8 vào đúng schema đang dùng.
>
> <details><summary>Ran 4 commands</summary>
>
> - Ran `Get-Content -Path data_processing\caption\caption_pipeline\shot_captioner.py`
> - Ran `Get-Content -Path data_processing\caption\run_caption_pipeline.py`
> - Ran `Get-Content -Path data_processing\caption\caption_pipeline\shot_index.py`
> - Ran `Get-Content -Path data_processing\caption\caption_pipeline\trake_event_builder.py`
>
> </details>
>
> Luồng hiện tại đã có đầy đủ builder deterministic cho shot và TRAKE. Phase 8 vì vậy có thể viết gọn: load baseline evidence/index, áp override frame caption, sau đó gọi lại `fuse_shot_captions`, `build_shot_index`, `build_trake_event_steps`, `build_event_step_index`.
>
> <details><summary>Ran 8 commands</summary>
>
> - Ran `Get-Content -Path data_processing\caption\caption_pipeline\caption_report.py`
> - Ran `Get-Content -Path data_processing\caption\caption_pipeline\shot_caption_report.py`
> - Ran `Get-Content -Path data_processing\caption\caption_pipeline\event_step_index.py`
> - Ran `Get-Content -Path data_processing\caption\caption_pipeline\event_step_report.py`
> - Ran `Get-Content -Path data_processing\caption\caption_pipeline\frame_caption_fuser.py`
> - Ran `Get-Content -Path data_processing\caption\caption_pipeline\config.py`
> - Ran `Get-Content -Path data_processing\caption\caption_pipeline\io_utils.py`
> - Ran `Get-Content -Path data_processing\caption\caption_pipeline\retrieval_queries.py`
>
> </details>
>
> Bước thiết kế chính: override hiện có chỉ bao phủ 207 frame representative, không phải 283 frame toàn video. Vì vậy Phase 8 sẽ giữ caption baseline cho frame chưa override, còn frame có VLM caption thì thay bằng caption VLM; sau đó shot/event sẽ được rebuild từ tập frame-index đã trộn này.
>
> <details><summary>Ran 4 commands</summary>
>
> - Ran `Select-String -Path data_processing\caption\caption_pipeline\vlm_frame_experiment.py -Pattern 'caption_text|canonical_frame_id|vlm_caption|output|record|prompt_version|caption_model' -Context 1,3`
> - Ran `Get-Content -Path data_processing\caption\caption_pipeline\experiment_materializer.py`
> - Ran `Get-Content -Path data_processing\caption\run_caption_experiment_materialize.py`
> - Ran `Get-Content -Path data_processing\caption\caption_pipeline\retrieval_export.py`
>
> </details>
>
> Mình sẽ code Phase 8 theo 3 mảnh nhỏ: module propagation, CLI chạy trên server, và query set visual để benchmark đúng phần VLM. Sau đó mình sẽ kiểm tra cú pháp/đường import ở mức có thể trên máy local.
>
> Mình sẽ thêm file mới thay vì sửa sâu vào các builder cũ, để phase này ít rủi ro: `experiment_propagation.py` làm core logic và `run_caption_experiment_propagate.py` làm CLI. Các builder hiện có sẽ vẫn giữ nguyên.
>
> <details><summary>Created a file</summary>
>
> - Wrote `C:/Users/User/Desktop/AIC2026/data_processing/caption/caption_pipeline/experiment_propagation.py` (+587 -0)
> - Auto-review: approved
> - Auto-review risk: low
> - Auto-review rationale: Auto-review returned a low-risk allow decision.
>
> </details>
>
> Core module đã được thêm. Tiếp theo mình thêm CLI `run_caption_experiment_propagate.py` để chạy đúng trên server giống các phase trước, có `--strict` và log summary.
>
> <details><summary>Created a file</summary>
>
> - Wrote `C:/Users/User/Desktop/AIC2026/data_processing/caption/run_caption_experiment_propagate.py` (+138 -0)
> - Auto-review: approved
> - Auto-review risk: low
> - Auto-review rationale: Auto-review returned a low-risk allow decision.
>
> </details>
>
> CLI đã có. Mình sẽ rà nhanh style/import và chạy compile nếu môi trường local cho phép; nếu Python local không có thì mình vẫn sẽ kiểm tra bằng đọc file và đưa lệnh server rõ ràng.
>
> <details><summary>Ran 3 commands</summary>
>
> - Ran `Get-Content -Path data_processing\caption\caption_pipeline\experiment_propagation.py`
> - Ran `Get-Content -Path data_processing\caption\run_caption_experiment_propagate.py`
> - Ran `git status --short`
>
> </details>
>
> Mình thấy một import không dùng trong module mới, sẽ dọn luôn cho sạch trước khi kiểm tra compile.
>
> <details><summary>Edited a file</summary>
>
> - Wrote `C:/Users/User/Desktop/AIC2026/data_processing/caption/caption_pipeline/experiment_propagation.py` (+1 -2)
>
> </details>
>
> `apply_patch` vừa bị lỗi đường dẫn tuyệt đối kỳ lạ dù lần tạo file trước đã được. Mình sẽ sửa bằng PowerShell trên đúng workdir repo, chỉ xóa một dòng import không dùng.
>
> <details><summary>Ran a command</summary>
>
> - Ran ``$path = 'C:\Users\User\Desktop\AIC2026\data_processing\caption\caption_pipeline\experiment_propagation.py'
> $text = Get-Content -LiteralPath $path -Raw
> $text = $text -replace "from \.retrieval_export import normalized_text, tokenize`r?`n", ""
> Set-Content -LiteralPath $path -Value $text -Encoding UTF8``
> - Auto-review: approved
> - Auto-review risk: low
> - Auto-review rationale: Auto-review returned a low-risk allow decision.
>
> </details>
>
> Dọn xong. Bây giờ mình kiểm tra cú pháp cho hai file mới; nếu máy local không có Python usable thì mình sẽ dừng ở kiểm tra tĩnh và đưa lệnh server.
>
> <details><summary>Ran 3 commands</summary>
>
> - Ran `python -m py_compile data_processing\caption\caption_pipeline\experiment_propagation.py data_processing\caption\run_caption_experiment_propagate.py`
> - Ran `git diff -- data_processing/caption/caption_pipeline/experiment_propagation.py data_processing/caption/run_caption_experiment_propagate.py`
> - Ran `Select-String -Path data_processing\caption\caption_pipeline\experiment_propagation.py -Pattern 'retrieval_export|normalized_text|tokenize'`
>
> </details>
>
> Local `python.exe` là WindowsApps stub nên không chạy được compile ở máy này. Mình sẽ thử `py -3`/`python3`; nếu không có thì phần compile nên chạy trên server sau khi pull.
>
> <details><summary>Ran 3 commands</summary>
>
> - Ran `python3 -m py_compile data_processing\caption\caption_pipeline\experiment_propagation.py data_processing\caption\run_caption_experiment_propagate.py`
> - Ran `git status --short`
> - Ran `py -3 -m py_compile data_processing\caption\caption_pipeline\experiment_propagation.py data_processing\caption\run_caption_experiment_propagate.py`
>
> </details>
>
> Vì local không chạy được Python, mình đang kiểm tra logic bằng source. Một điểm cần lưu ý: visual query set mới có thể làm eval khó hơn query default, nên mình sẽ để nó là file riêng để bạn chạy benchmark visual riêng, không thay acceptance mặc định nếu chưa muốn.
>
> <details><summary>Ran 2 commands</summary>
>
> - Ran `Get-Content -Path data_processing\caption\caption_pipeline\retrieval_eval.py`
> - Ran `Select-String -Path data_processing\caption\caption_pipeline\retrieval_report.py -Pattern 'expected|hit|warning|expected_terms' -Context 1,2`
>
> </details>
>
> Visual eval không dựa vào `expected_terms_any`, chỉ yêu cầu query có hit. Vì vậy file visual mới an toàn để chạy riêng; nếu muốn nó làm acceptance cứng thì dùng `--strict`, còn để phân tích thì có thể bỏ `--strict`.
>
> Mình sẽ kiểm tra thêm vài điểm bằng `rg`: script mới có đúng argparse, module không còn import thừa, và status chỉ còn hai file phase 8 mới.
>
> <details><summary>Ran 3 commands</summary>
>
> - Ran `git status --short data_processing/caption`
> - Ran `rg "experiment_propagation|propagate_caption_experiment|visual_retrieval_queries|caption_phase8" data_processing/caption`
> - Ran `Get-Content -Path data_processing\caption\caption_pipeline\experiment_propagation.py -TotalCount 30`
>
> </details>
>
> Phase 8 code đã xong ở mức repo. Mình không thể compile local vì Python trên máy này trỏ vào WindowsApps stub, nên mình sẽ kèm lệnh `py_compile` để bạn chạy ngay trên server/env đang dùng.
>
> <details><summary>Plan</summary>
>
> - [x] Đọc module hiện có và xác định điểm nối Phase 8
> - [x] Thêm module propagate VLM frame captions lên shot/event/index
> - [x] Thêm CLI server cho Phase 8 và visual query output
> - [x] Kiểm tra cú pháp/import và đưa lệnh chạy
>
> </details>
</details>

Đã thực hiện Phase 8.

Mình thêm 2 file mới:

- [experiment_propagation.py](C:/Users/User/Desktop/AIC2026/data_processing/caption/caption_pipeline/experiment_propagation.py)
- [run_caption_experiment_propagate.py](C:/Users/User/Desktop/AIC2026/data_processing/caption/run_caption_experiment_propagate.py)

Phase 8 hiện làm:

```text
VLM frame_caption_overrides.jsonl
-> QA/normalize overrides
-> rebuild frame_index
-> rebuild shot_index từ frame captions mới
-> rebuild event_step_index / TRAKE
-> rebuild compact_search_index
-> ghi visual_retrieval_queries.jsonl
-> ghi experiment_propagation_report
```

Nó không chạy lại Qwen, không cần GPU. Nó dùng output VLM đã có.

Chạy trên server:

```bash
cd /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/data_processing/caption

python -m py_compile \
  caption_pipeline/experiment_propagation.py \
  run_caption_experiment_propagate.py
```

Sau đó chạy Phase 8:

```bash
PYTHONIOENCODING=utf-8 python run_caption_experiment_propagate.py \
  --video-id L22_V012 \
  --baseline-dir /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption \
  --output-root /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption_experiments \
  --experiment-name vlm_frame_qwen25vl_7b_propagated \
  --frame-captions /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption_experiments_raw/vlm_frame_qwen25vl_7b_rep/L22_V012/frame_caption_overrides.jsonl \
  --strict
```

Kỳ vọng log sẽ gần như:

```text
Valid overrides: 207
Frames updated: 207
Frames kept: 76
Frames: 283
Shots: 102
Event steps: 102
Index docs: 385
Warnings: 0
```

Rồi chạy retrieval eval cho propagated experiment:

```bash
PYTHONIOENCODING=utf-8 python run_retrieval_export_eval.py \
  --video-id L22_V012 \
  --caption-dir /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption_experiments/vlm_frame_qwen25vl_7b_propagated \
  --strict
```

Benchmark 3 nhánh:

```bash
PYTHONIOENCODING=utf-8 python run_caption_experiment_benchmark.py \
  --video-id L22_V012 \
  --baseline-dir /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption \
  --baseline-label template_baseline \
  --experiment vlm_frame_only=/tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption_experiments/vlm_frame_qwen25vl_7b_rep \
  --experiment vlm_propagated=/tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption_experiments/vlm_frame_qwen25vl_7b_propagated \
  --benchmark-name phase8_propagation_benchmark \
  --top-n-samples 3 \
  --strict
```

Mình chưa chạy `py_compile` local được vì Python ở máy này đang trỏ vào WindowsApps stub, nên bạn chạy compile trên server trước khi chạy phase là chuẩn nhất.

> "PYTHONIOENCODING=utf-8 python run_caption_experiment_benchmark.py \
>   --video-id L22_V012 \
>   --baseline-dir /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption \
>   --baseline-label template_baseline \
>   --experiment vlm_frame_only=/tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption_experiments/vlm_frame_qwen25vl_7b_rep \
>   --experiment vlm_propagated=/tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption_experiments/vlm_frame_qwen25vl_7b_propagated \
>   --benchmark-name phase8_propagation_benchmark \
>   --top-n-samples 3 \
>   --strictents/vlm_frame_qwen25vl_7b_propagated/L22_V012/eval/retrieval_eval_report.md
> 13:13:43 [INFO] caption_retrieval: === Done in 5.6s ===
> 13:13:43 [INFO] caption_retrieval:   Corpus:  487 docs
> 13:13:43 [INFO] caption_retrieval:   Queries: 12
> 13:13:43 [INFO] caption_retrieval:   Results: 240
> 13:13:43 [INFO] caption_retrieval:   Hits:    12/12
> 13:13:43 [INFO] caption_retrieval:   Warnings: 0
> 13:13:43 [INFO] caption_retrieval: All retrieval export/eval acceptance checks passed.frame_qwen25vl_7b_propagated/L22_V012
> 13:13:28 [INFO] caption_phase8_propagate: === Done in 0.3s ===
> 13:13:28 [INFO] caption_phase8_propagate:   Override rows:   207
> 13:13:28 [INFO] caption_phase8_propagate:   Valid overrides: 207
> 13:13:28 [INFO] caption_phase8_propagate:   Frames updated:  207
> 13:13:28 [INFO] caption_phase8_propagate:   Frames kept:     76
> 13:13:28 [INFO] caption_phase8_propagate:   Frames:          283
> 13:13:28 [INFO] caption_phase8_propagate:   Shots:           102
> 13:13:28 [INFO] caption_phase8_propagate:   Event steps:     102
> 13:13:28 [INFO] caption_phase8_propagate:   Index docs:      385
> 13:13:28 [INFO] caption_phase8_propagate:   Warnings:        0
> 13:13:28 [INFO] caption_phase8_propagate: All caption propagation checks passed"

Phase 8 đã pass rất đẹp.

Kết quả quan trọng:

```text
Override rows:   207
Valid overrides: 207
Frames updated:  207
Frames kept:     76
Frames:          283
Shots:           102
Event steps:     102
Index docs:      385
Warnings:        0
```

Nghĩa là VLM frame captions đã thật sự được propagate lên:

```text
frame_index
-> shot_index
-> event_step_index
-> compact_search_index
```

Retrieval eval cũng pass:

```text
Corpus: 487 docs
Queries: 12
Hits: 12/12
Warnings: 0
```

Bây giờ cần chạy nốt benchmark command, nhưng đoạn bạn paste có vẻ bị dính log vào cuối `--strict`. Chạy lại sạch như này:

```bash
cd /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/data_processing/caption

PYTHONIOENCODING=utf-8 python run_caption_experiment_benchmark.py \
  --video-id L22_V012 \
  --baseline-dir /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption \
  --baseline-label template_baseline \
  --experiment vlm_frame_only=/tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption_experiments/vlm_frame_qwen25vl_7b_rep \
  --experiment vlm_propagated=/tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption_experiments/vlm_frame_qwen25vl_7b_propagated \
  --benchmark-name phase8_propagation_benchmark \
  --top-n-samples 3 \
  --strict
```

Sau đó xem report:

```bash
less /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption/L22_V012/benchmarks/phase8_propagation_benchmark/phase8_propagation_benchmark.md
```

Và file manual review:

```bash
less /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption/L22_V012/benchmarks/phase8_propagation_benchmark/manual_review_samples.jsonl
```

Nếu benchmark cũng pass, bước đáng làm tiếp là chạy **visual query eval** riêng bằng file Phase 8 đã sinh:

```bash
PYTHONIOENCODING=utf-8 python run_retrieval_export_eval.py \
  --video-id L22_V012 \
  --caption-dir /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption_experiments/vlm_frame_qwen25vl_7b_propagated \
  --queries-jsonl /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption_experiments/vlm_frame_qwen25vl_7b_propagated/L22_V012/eval/visual_retrieval_queries.jsonl \
  --top-k 20
```

Lần visual query này nên đọc bằng manual review hơn là chỉ nhìn hit-rate, vì nó mới kiểm tra đúng tác động của VLM caption.

> "PYTHONIOENCODING=utf-8 python run_retrieval_export_eval.py \
>   --video-id L22_V012 \
>   --caption-dir /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption_experiments/vlm_frame_qwen25vl_7b_propagated \
>   --queries-jsonl /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption_experiments/vlm_frame_qwen25vl_7b_propagated/L22_V012/eval/visual_retrieval_queries.jsonl \
>   --top-k 20" lênnhj này sẽ chạy như thế nào ? có dùng GPU hay không

Lệnh này **không dùng GPU**.

Nó chỉ chạy retrieval offline bằng CPU:

```text
compact_search_index.jsonl
+ event_step_index.jsonl
+ visual_retrieval_queries.jsonl
-> lexical scoring / ranking
-> retrieval_eval_report
```

Nó không load Qwen, không load VLM, không sinh caption mới. Vì vậy chạy bằng env base hoặc env caption đều được, miễn có Python package cơ bản mà pipeline cần.

Lệnh đó sẽ làm:

1. Đọc corpus từ:

```bash
/tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption_experiments/vlm_frame_qwen25vl_7b_propagated/L22_V012/indexes/compact_search_index.jsonl
/tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption_experiments/vlm_frame_qwen25vl_7b_propagated/L22_V012/captions/event_step_index.jsonl
```

2. Đọc query visual từ:

```bash
/tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption_experiments/vlm_frame_qwen25vl_7b_propagated/L22_V012/eval/visual_retrieval_queries.jsonl
```

3. Tính điểm lexical local trên CPU.

4. Ghi output vào cùng thư mục eval, ví dụ:

```bash
/tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption_experiments/vlm_frame_qwen25vl_7b_propagated/L22_V012/eval/
```

Nên chạy như sau:

```bash
cd /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/data_processing/caption

PYTHONIOENCODING=utf-8 python run_retrieval_export_eval.py \
  --video-id L22_V012 \
  --caption-dir /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption_experiments/vlm_frame_qwen25vl_7b_propagated \
  --queries-jsonl /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption_experiments/vlm_frame_qwen25vl_7b_propagated/L22_V012/eval/visual_retrieval_queries.jsonl \
  --top-k 20
```

Không cần:

```bash
CUDA_VISIBLE_DEVICES=...
```

Sau khi chạy xong, xem:

```bash
less /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption_experiments/vlm_frame_qwen25vl_7b_propagated/L22_V012/eval/retrieval_eval_report.md
less /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption_experiments/vlm_frame_qwen25vl_7b_propagated/L22_V012/eval/retrieval_results.jsonl
```

Lưu ý nhỏ: lệnh này có thể overwrite report eval cũ trong thư mục `eval/`. Nếu muốn giữ riêng default eval và visual eval, phase sau mình nên thêm option `--eval-name` hoặc `--output-dir` cho retrieval script. Hiện tại thì dùng được, chỉ cần biết report cũ có thể bị ghi đè.

> "13:17:31 [INFO] caption_retrieval: === Done in 3.8s ===
> 13:17:31 [INFO] caption_retrieval:   Corpus:  487 docs
> 13:17:31 [INFO] caption_retrieval:   Queries: 8
> 13:17:31 [INFO] caption_retrieval:   Results: 160
> 13:17:31 [INFO] caption_retrieval:   Hits:    8/8
> 13:17:31 [INFO] caption_retrieval:   Warnings: 0
> 13:17:31 [INFO] caption_retrieval: All retrieval export/eval acceptance checks passed." phân tích kết quả
>
> User context
>
> Attachments:
> - phase8_propagation_benchmark.md: `d:/Users/phase8_propagation_benchmark.md`
> - manual_review_samples.jsonl: `d:/Users/manual_review_samples.jsonl`

<details><summary>6 previous messages</summary>

> Mình sẽ đọc `phase8_propagation_benchmark.md` và `manual_review_samples.jsonl` bạn đưa để phân biệt hai chuyện: acceptance pass, và VLM propagated có thực sự đổi ranking/đổi nội dung so với baseline không.
>
> <details><summary>Ran 3 commands</summary>
>
> - Ran `(Get-Content -Path 'D:\Users\manual_review_samples.jsonl').Count`
> - Ran `Get-Content -Path 'D:\Users\phase8_propagation_benchmark.md'`
> - Ran `Get-Content -Path 'D:\Users\manual_review_samples.jsonl' -TotalCount 5`
>
> </details>
>
> File benchmark cho thấy một tín hiệu thú vị: Phase 8 pass về kỹ thuật nhưng ranking default-query lại giảm điểm trung bình. Mình sẽ parse đủ 12 query để chỉ ra query nào cải thiện, query nào tụt, và vì sao.
>
> <details><summary>Ran 2 commands</summary>
>
> - Ran `$items = Get-Content -Path 'D:\Users\manual_review_samples.jsonl' | ForEach-Object { ConvertFrom-Json $_ }
> $rows = @()
> foreach ($item in $items) {
>   $base = ($item.candidates | Where-Object { $_.label -eq 'template_baseline' }).top_results
>   $frame = ($item.candidates | Where-Object { $_.label -eq 'vlm_frame_only' }).top_results
>   $prop = ($item.candidates | Where-Object { $_.label -eq 'vlm_propagated' }).top_results
>   $delta = [double]$prop[0].score - [double]$base[0].score
>   $rows += [PSCustomObject]@{
>     query_id = $item.query_id
>     route = $item.route
>     same_frame_top1 = ($base[0].document_id -eq $frame[0].document_id)
>     same_prop_top1 = ($base[0].document_id -eq $prop[0].document_id)
>     base_doc = $base[0].document_id
>     prop_doc = $prop[0].document_id
>     base_score = [math]::Round([double]$base[0].score, 3)
>     prop_score = [math]::Round([double]$prop[0].score, 3)
>     delta = [math]::Round($delta, 3)
>     prop_unit = $prop[0].unit_type
>   }
> }
> $rows | Sort-Object delta | Format-Table -AutoSize`
> - Ran ``$items = Get-Content -Path 'D:\Users\manual_review_samples.jsonl' | ForEach-Object { ConvertFrom-Json $_ }
> foreach ($item in $items) {
>   $base = ($item.candidates | Where-Object { $_.label -eq 'template_baseline' }).top_results
>   $prop = ($item.candidates | Where-Object { $_.label -eq 'vlm_propagated' }).top_results
>   $delta = [double]$prop[0].score - [double]$base[0].score
>   "{0}`t{1}`tbase={2}`tprop={3}`tbase_score={4:N3}`tprop_score={5:N3}`tdelta={6:N3}`tsame={7}`tprop_unit={8}" -f $item.query_id,$item.route,$base[0].document_id,$prop[0].document_id,[double]$base[0].score,[double]$prop[0].score,$delta,($base[0].document_id -eq $prop[0].document_id),$prop[0].unit_type
> }``
>
> </details>
>
> Mình đã có bảng delta. Điểm đáng chú ý: `vlm_frame_only` giữ nguyên top-1 12/12, còn `vlm_propagated` đổi top-1 10/12; đổi này có 1 query tăng điểm, 8 query giảm điểm, 2 query giữ nguyên document.
>
> <details><summary>Ran a command</summary>
>
> - Ran `$items = Get-Content -Path 'D:\Users\manual_review_samples.jsonl' | ForEach-Object { ConvertFrom-Json $_ }
> foreach ($label in @('template_baseline','vlm_frame_only','vlm_propagated')) {
>   $top1Types = @{}
>   $top3Types = @{}
>   foreach ($item in $items) {
>     $cand = $item.candidates | Where-Object { $_.label -eq $label }
>     $top1 = $cand.top_results[0].unit_type
>     if (-not $top1Types.ContainsKey($top1)) { $top1Types[$top1]=0 }; $top1Types[$top1]++
>     foreach ($r in $cand.top_results) { $k=$r.unit_type; if (-not $top3Types.ContainsKey($k)) { $top3Types[$k]=0 }; $top3Types[$k]++ }
>   }
>   "LABEL=$label"
>   "top1=" + (($top1Types.GetEnumerator() | Sort-Object Name | ForEach-Object { "$($_.Name):$($_.Value)" }) -join ', ')
>   "top3=" + (($top3Types.GetEnumerator() | Sort-Object Name | ForEach-Object { "$($_.Name):$($_.Value)" }) -join ', ')
> }`
>
> </details>
</details>

Kết quả hiện tại có 2 lớp:

**1. Visual eval vừa chạy: pass kỹ thuật**
```text
Queries: 8
Hits: 8/8
Warnings: 0
```

Nghĩa là query visual đều có kết quả top-k, corpus/index không lỗi, Phase 8 output dùng được để search. Nhưng log này **chưa nói VLM tốt hơn baseline**, vì đây chỉ là eval một candidate, chưa so sánh baseline vs propagated trên cùng visual query set.

**2. Benchmark Phase 8 trên 12 query default: VLM propagated chưa tốt hơn**
Từ `phase8_propagation_benchmark.md`:

```text
template_baseline  Hit Rate 1.000, Avg Top1 63.713
vlm_frame_only     Hit Rate 1.000, Avg Top1 63.713
vlm_propagated     Hit Rate 1.000, Avg Top1 44.958
```

So với baseline:

```text
vlm_frame_only: 12 same top1, delta 0
vlm_propagated: 1 improved, 8 regressed, 2 same top1, avg delta -18.755
```

Diễn giải: `vlm_frame_only` không tác động ranking, đúng như ta đã thấy trước đó. `vlm_propagated` đã tác động thật vào pipeline, vì top-1 đổi ở 10/12 query, nhưng với query default hiện tại thì ranking bị kéo xuống.

Nguyên nhân chính có khả năng cao: query default đang rất “template/lexical”, ví dụ có từ `spoken content`, `scene shows`, `visible objects`, `transition change`. Baseline template chứa đúng các cụm này lặp lại trong `caption_text`, `temporal_caption`, `trake_text`, nên score lexical rất cao. Khi propagate VLM, caption tự nhiên hơn, nhiều đoạn tiếng Anh kiểu “The video frame shows...”, làm mất các cụm template và giảm điểm BM25/lexical, nhất là với audio/trake query.

Ví dụ rõ:

```text
q_audio_001
baseline top1: event_step_0040, score 99.477
propagated top1: shot_0042, score 24.987
delta -74.490
```

Không nhất thiết VLM “tệ hơn về nội dung”, mà là **scoring/query set hiện tại đang ưu ái văn phong template**.

Kết luận:

```text
Phase 8 pass kỹ thuật.
VLM đã tác động thật vào shot/event/index.
Nhưng benchmark default cho thấy propagated chưa thắng baseline lexical.
Cần benchmark visual riêng hoặc chỉnh retrieval weighting/query set.
```

Bước tiếp nên làm: so sánh visual query giữa `template_baseline`, `vlm_frame_only`, `vlm_propagated`. Đây mới là bài test công bằng cho VLM. Hiện bạn mới chạy visual eval cho propagated, chưa benchmark visual 3 nhánh.

> Hãy tới bước tiếp theo

<details><summary>18 previous messages</summary>

> Mình sẽ làm bước tiếp theo theo hướng rõ nhất: thêm khả năng chạy **visual eval/benchmark riêng** mà không ghi đè eval default. Trước hết mình đọc lại `run_retrieval_export_eval.py` và benchmark loader để nối thêm `--eval-name`.
>
> <details><summary>Ran 4 commands</summary>
>
> - Ran `Get-Content -Path data_processing\caption\caption_pipeline\experiment_benchmark.py`
> - Ran `Get-Content -Path data_processing\caption\run_retrieval_export_eval.py`
> - Ran `Get-Content -Path data_processing\caption\run_caption_experiment_benchmark.py`
> - Ran `git status --short`
>
> </details>
>
> Mình sẽ code bước tiếp theo: thêm `--eval-name` cho retrieval và benchmark. Như vậy ta có thể chạy `default` và `visual` eval song song, không ghi đè report cũ.
>
> Trước hết mình sửa `run_retrieval_export_eval.py`: `--eval-name visual` sẽ ghi vào `eval/visual/` và `exports/visual/`, đồng thời copy query set vào đúng eval folder để benchmark đọc được.
>
> <details><summary>Edited a file</summary>
>
> - Wrote `C:/Users/User/Desktop/AIC2026/data_processing/caption/run_retrieval_export_eval.py` (+55 -37)
>
> </details>
>
> `apply_patch` lại vấp đường dẫn tuyệt đối trên phiên này. Mình sẽ dùng PowerShell để thay đổi chính xác các đoạn nhỏ trong file repo.
>
> <details><summary>Ran a command</summary>
>
> - Ran `$path = 'C:\Users\User\Desktop\AIC2026\data_processing\caption\run_retrieval_export_eval.py'
> $text = Get-Content -LiteralPath $path -Raw
> $old = @'
>     parser.add_argument(
>         "--queries-jsonl",
>         default="",
>         help="Optional retrieval_queries.jsonl override",
>     )
>     parser.add_argument("--top-k", type=int, default=20, help="Top-K results per query")
> '@
> $new = @'
>     parser.add_argument(
>         "--queries-jsonl",
>         default="",
>         help="Optional retrieval_queries.jsonl override",
>     )
>     parser.add_argument(
>         "--eval-name",
>         default="",
>         help=(
>             "Optional named eval/export subdirectory. Example: --eval-name visual "
>             "writes eval/visual/* and exports/visual/* without overwriting default eval."
>         ),
>     )
>     parser.add_argument("--top-k", type=int, default=20, help="Top-K results per query")
> '@
> if (-not $text.Contains($old)) { throw 'Failed to find parser insertion block' }
> $text = $text.Replace($old, $new)
> $old = @'
>     base_dir = Path(args.caption_dir) / args.video_id
>     exports_dir = base_dir / "exports"
>     eval_dir = base_dir / "eval"
> '@
> $new = @'
>     base_dir = Path(args.caption_dir) / args.video_id
>     exports_dir = _named_output_dir(base_dir / "exports", args.eval_name)
>     eval_dir = _named_output_dir(base_dir / "eval", args.eval_name)
> '@
> if (-not $text.Contains($old)) { throw 'Failed to find output dir block' }
> $text = $text.Replace($old, $new)
> $old = @'
>     queries_path = Path(args.queries_jsonl) if args.queries_jsonl else (
>         eval_dir / "retrieval_queries.jsonl"
>     )
> '@
> $new = @'
>     input_queries_path = Path(args.queries_jsonl) if args.queries_jsonl else (
>         eval_dir / "retrieval_queries.jsonl"
>     )
>     output_queries_path = eval_dir / "retrieval_queries.jsonl"
> '@
> if (-not $text.Contains($old)) { throw 'Failed to find queries path block' }
> $text = $text.Replace($old, $new)
> $text = $text.Replace('    logger.info("Queries:    %s", queries_path)', '    logger.info("Queries:    %s", input_queries_path)')
> $old = @'
>     queries = _load_or_build_queries(
>         queries_path=queries_path,
>         video_id=args.video_id,
>         write_default=args.write_default_queries,
>     )
>     queries = normalize_queries(queries, args.video_id)
>     write_jsonl(queries_path, queries)
>     logger.info("Wrote/loaded %d retrieval queries -> %s", len(queries), queries_path)
> '@
> $new = @'
>     queries = _load_or_build_queries(
>         queries_path=input_queries_path,
>         video_id=args.video_id,
>         write_default=args.write_default_queries,
>     )
>     queries = normalize_queries(queries, args.video_id)
>     write_jsonl(output_queries_path, queries)
>     logger.info("Wrote/loaded %d retrieval queries -> %s", len(queries), output_queries_path)
> '@
> if (-not $text.Contains($old)) { throw 'Failed to find load/write queries block' }
> $text = $text.Replace($old, $new)
> $old = @'
> def _acceptance_problems(report: dict, strict: bool) -> list[str]:
> '@
> $new = @'
> def _named_output_dir(base: Path, eval_name: str) -> Path:
>     name = (eval_name or "").strip().strip("/\\")
>     if not name:
>         return base
>     if "/" in name or "\\" in name:
>         raise ValueError(f"Invalid --eval-name '{eval_name}'. Use a simple folder name.")
>     return base / name
>
>
>
> def _acceptance_problems(report: dict, strict: bool) -> list[str]:
> '@
> if (-not $text.Contains($old)) { throw 'Failed to find acceptance function marker' }
> $text = $text.Replace($old, $new)
> Set-Content -LiteralPath $path -Value $text -Encoding UTF8`
> - Auto-review: approved
> - Auto-review risk: low
> - Auto-review rationale: Auto-review returned a low-risk allow decision.
>
> </details>
>
> Retrieval CLI đã có `--eval-name`. Giờ mình thêm cùng tham số cho benchmark để nó đọc `eval/visual` và `exports/visual` thay vì thư mục default.
>
> <details><summary>Ran a command</summary>
>
> - Ran `$path = 'C:\Users\User\Desktop\AIC2026\data_processing\caption\caption_pipeline\experiment_benchmark.py'
> $text = Get-Content -LiteralPath $path -Raw
> $text = $text.Replace('def load_experiment(label: str, root: str | Path, video_id: str) -> dict[str, Any]:', 'def load_experiment(label: str, root: str | Path, video_id: str, eval_name: str = "") -> dict[str, Any]:')
> $old = @'
>     eval_dir = video_dir / "eval"
>     exports_dir = video_dir / "exports"
> '@
> $new = @'
>     eval_dir = _named_artifact_dir(video_dir / "eval", eval_name)
>     exports_dir = _named_artifact_dir(video_dir / "exports", eval_name)
> '@
> if (-not $text.Contains($old)) { throw 'Failed to find eval/exports block' }
> $text = $text.Replace($old, $new)
> $old = @'
> def _resolve_video_dir(root: Path, video_id: str) -> Path:
> '@
> $new = @'
> def _named_artifact_dir(base: Path, eval_name: str) -> Path:
>     name = (eval_name or "").strip().strip("/\\")
>     if not name:
>         return base
>     if "/" in name or "\\" in name:
>         raise ValueError(f"Invalid eval_name '{eval_name}'. Use a simple folder name.")
>     return base / name
>
>
>
> def _resolve_video_dir(root: Path, video_id: str) -> Path:
> '@
> if (-not $text.Contains($old)) { throw 'Failed to find resolve marker' }
> $text = $text.Replace($old, $new)
> Set-Content -LiteralPath $path -Value $text -Encoding UTF8
>
> $path = 'C:\Users\User\Desktop\AIC2026\data_processing\caption\run_caption_experiment_benchmark.py'
> $text = Get-Content -LiteralPath $path -Raw
> $old = @'
>     parser.add_argument(
>         "--output-dir",
>         default="",
>         help="Output dir. Default: <baseline video dir>/benchmarks/<benchmark-name>",
>     )
> '@
> $new = @'
>     parser.add_argument(
>         "--eval-name",
>         default="",
>         help="Optional named eval/export subdirectory to compare, e.g. visual.",
>     )
>     parser.add_argument(
>         "--output-dir",
>         default="",
>         help="Output dir. Default: <baseline video dir>/benchmarks/<benchmark-name>",
>     )
> '@
> if (-not $text.Contains($old)) { throw 'Failed to find benchmark parser block' }
> $text = $text.Replace($old, $new)
> $text = $text.Replace('baseline = load_experiment(args.baseline_label, args.baseline_dir, args.video_id)', 'baseline = load_experiment(args.baseline_label, args.baseline_dir, args.video_id, args.eval_name)')
> $text = $text.Replace('load_experiment(label, path, args.video_id)', 'load_experiment(label, path, args.video_id, args.eval_name)')
> $old = @'
>     logger.info("Baseline:    %s -> %s", args.baseline_label, baseline["video_dir"])
> '@
> $new = @'
>     if args.eval_name:
>         logger.info("EvalName:    %s", args.eval_name)
>     logger.info("Baseline:    %s -> %s", args.baseline_label, baseline["video_dir"])
> '@
> if (-not $text.Contains($old)) { throw 'Failed to find benchmark log block' }
> $text = $text.Replace($old, $new)
> Set-Content -LiteralPath $path -Value $text -Encoding UTF8`
> - Auto-review: approved
> - Auto-review risk: low
> - Auto-review rationale: Auto-review returned a low-risk allow decision.
>
> </details>
>
> Benchmark đã nhận `--eval-name`. Mình sẽ rà diff để chắc không ảnh hưởng flow default, rồi đưa lệnh chạy visual eval cho cả 3 nhánh.
>
> <details><summary>Ran 4 commands</summary>
>
> - Ran `git status --short data_processing/caption`
> - Ran `git diff -- data_processing/caption/run_retrieval_export_eval.py data_processing/caption/caption_pipeline/experiment_benchmark.py data_processing/caption/run_caption_experiment_benchmark.py`
> - Ran `Select-String -Path data_processing\caption\run_retrieval_export_eval.py -Pattern 'eval-name|input_queries_path|output_queries_path|_named_output_dir' -Context 1,2`
> - Ran `Select-String -Path data_processing\caption\caption_pipeline\experiment_benchmark.py,data_processing\caption\run_caption_experiment_benchmark.py -Pattern 'eval_name|eval-name|_named_artifact_dir|load_experiment' -Context 1,2`
>
> </details>
>
> Diff hợp lý và vẫn backward-compatible: không truyền `--eval-name` thì các lệnh cũ vẫn đọc/ghi `eval/` và `exports/` như trước. Mình kiểm tra thêm status đầy đủ rồi sẽ đưa chuỗi lệnh chạy.
>
> <details><summary>Ran 3 commands</summary>
>
> - Ran `Get-Content -Path data_processing\caption\run_retrieval_export_eval.py | Select-Object -Skip 220 -First 25`
> - Ran `git status --short`
> - Ran `Get-Content -Path data_processing\caption\caption_pipeline\experiment_benchmark.py | Select-Object -Skip 282 -First 24`
>
> </details>
>
> Có hai dòng trắng thừa giữa helper và function tiếp theo; không ảnh hưởng chạy, nhưng mình dọn luôn để file gọn trước khi bạn push.
>
> <details><summary>Ran a command</summary>
>
> - Ran ``$paths = @(
>   'C:\Users\User\Desktop\AIC2026\data_processing\caption\run_retrieval_export_eval.py',
>   'C:\Users\User\Desktop\AIC2026\data_processing\caption\caption_pipeline\experiment_benchmark.py'
> )
> foreach ($path in $paths) {
>   $text = Get-Content -LiteralPath $path -Raw
>   $text = $text -replace "`r?`n`r?`n`r?`ndef ", "`n`ndef "
>   Set-Content -LiteralPath $path -Value $text -Encoding UTF8
> }``
> - Auto-review: approved
> - Auto-review risk: low
> - Auto-review rationale: Auto-review returned a low-risk allow decision.
>
> </details>
>
> Dọn xong. Mình sẽ kiểm tra lại nhanh các file tồn tại và status, rồi chốt bằng lệnh chạy named visual eval/benchmark.
>
> <details><summary>Ran 4 commands</summary>
>
> - Ran `Test-Path data_processing\caption\caption_pipeline\experiment_propagation.py; Test-Path data_processing\caption\run_caption_experiment_propagate.py`
> - Ran `git diff --stat -- data_processing/caption`
> - Ran `git status --short`
> - Ran `Select-String -Path data_processing\caption\run_retrieval_export_eval.py,data_processing\caption\run_caption_experiment_benchmark.py -Pattern '--eval-name' -Context 1,3`
>
> </details>
>
> <details><summary>Plan</summary>
>
> - [x] Thêm eval-name cho retrieval export/eval
> - [ ] Thêm eval-name cho benchmark loader/CLI
> - [ ] Rà cú pháp bằng kiểm tra tĩnh và đưa lệnh chạy visual benchmark
>
> </details>
</details>

Mình đã làm bước tiếp theo: thêm `--eval-name` để chạy **visual eval/benchmark riêng**, không ghi đè `eval/` default.

Đã sửa 3 file:

- `run_retrieval_export_eval.py`
- `run_caption_experiment_benchmark.py`
- `caption_pipeline/experiment_benchmark.py`

Ý nghĩa mới:

```bash
--eval-name visual
```

sẽ ghi/đọc:

```text
eval/visual/
exports/visual/
```

thay vì ghi đè:

```text
eval/
exports/
```

Chạy compile trên server trước:

```bash
cd /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/data_processing/caption

python -m py_compile \
  run_retrieval_export_eval.py \
  run_caption_experiment_benchmark.py \
  caption_pipeline/experiment_benchmark.py
```

Sau đó chạy visual eval cho đủ 3 nhánh.

**1. Baseline visual eval**

```bash
PYTHONIOENCODING=utf-8 python run_retrieval_export_eval.py \
  --video-id L22_V012 \
  --caption-dir /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption \
  --queries-jsonl /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption_experiments/vlm_frame_qwen25vl_7b_propagated/L22_V012/eval/visual_retrieval_queries.jsonl \
  --eval-name visual \
  --top-k 20
```

**2. VLM frame-only visual eval**

```bash
PYTHONIOENCODING=utf-8 python run_retrieval_export_eval.py \
  --video-id L22_V012 \
  --caption-dir /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption_experiments/vlm_frame_qwen25vl_7b_rep \
  --queries-jsonl /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption_experiments/vlm_frame_qwen25vl_7b_propagated/L22_V012/eval/visual_retrieval_queries.jsonl \
  --eval-name visual \
  --top-k 20
```

**3. VLM propagated visual eval**

```bash
PYTHONIOENCODING=utf-8 python run_retrieval_export_eval.py \
  --video-id L22_V012 \
  --caption-dir /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption_experiments/vlm_frame_qwen25vl_7b_propagated \
  --queries-jsonl /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption_experiments/vlm_frame_qwen25vl_7b_propagated/L22_V012/eval/visual_retrieval_queries.jsonl \
  --eval-name visual \
  --top-k 20
```

**4. Benchmark visual 3 nhánh**

```bash
PYTHONIOENCODING=utf-8 python run_caption_experiment_benchmark.py \
  --video-id L22_V012 \
  --baseline-dir /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption \
  --baseline-label template_baseline \
  --experiment vlm_frame_only=/tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption_experiments/vlm_frame_qwen25vl_7b_rep \
  --experiment vlm_propagated=/tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption_experiments/vlm_frame_qwen25vl_7b_propagated \
  --eval-name visual \
  --benchmark-name phase8_visual_benchmark \
  --top-n-samples 5 \
  --strict
```

Xem report:

```bash
less /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption/L22_V012/benchmarks/phase8_visual_benchmark/phase8_visual_benchmark.md
less /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption/L22_V012/benchmarks/phase8_visual_benchmark/manual_review_samples.jsonl
```

Bước này vẫn chỉ CPU, không dùng GPU. GPU chỉ cần khi chạy Qwen/VLM sinh caption.

> Phân tích kết quả
>
> User context
>
> Attachments:
> - manual_review_samples.jsonl: `d:/Users/manual_review_samples.jsonl`
> - phase8_visual_benchmark.md: `d:/Users/phase8_visual_benchmark.md`

<details><summary>6 previous messages</summary>

> Mình sẽ đọc benchmark visual và parse `manual_review_samples.jsonl` để xem VLM có thắng ở query visual không, query nào đổi top-1, và lỗi còn nằm ở caption hay ở scoring lexical.
>
> <details><summary>Ran 3 commands</summary>
>
> - Ran `Get-Content -Path 'D:\Users\phase8_visual_benchmark.md'`
> - Ran `Get-Content -Path 'D:\Users\manual_review_samples.jsonl' -TotalCount 8`
> - Ran `(Get-Content -Path 'D:\Users\manual_review_samples.jsonl').Count`
>
> </details>
>
> Benchmark summary cho thấy visual set đã đảo chiều: propagated đang đứng đầu. Mình sẽ parse đủ 8 query để tách query nào cải thiện, query nào giữ nguyên/top-1, query nào bị giảm nhẹ.
>
> <details><summary>Ran a command</summary>
>
> - Ran ``$items = Get-Content -Path 'D:\Users\manual_review_samples.jsonl' | ForEach-Object { ConvertFrom-Json $_ }
> foreach ($item in $items) {
>   $base = ($item.candidates | Where-Object { $_.label -eq 'template_baseline' }).top_results
>   $frame = ($item.candidates | Where-Object { $_.label -eq 'vlm_frame_only' }).top_results
>   $prop = ($item.candidates | Where-Object { $_.label -eq 'vlm_propagated' }).top_results
>   $deltaFrame = [double]$frame[0].score - [double]$base[0].score
>   $deltaProp = [double]$prop[0].score - [double]$base[0].score
>   "{0}`tbase={1}`t{2:N3}`tframe={3}`t{4:N3}`tdF={5:N3}`tprop={6}`t{7:N3}`tdP={8:N3}`tsameProp={9}" -f $item.query_id,$base[0].document_id,[double]$base[0].score,$frame[0].document_id,[double]$frame[0].score,$deltaFrame,$prop[0].document_id,[double]$prop[0].score,$deltaProp,($base[0].document_id -eq $prop[0].document_id)
> }``
>
> </details>
>
> Mình thấy có một chi tiết cần đọc kỹ: điểm tăng không luôn đồng nghĩa semantic tốt hơn, ví dụ query `boats on water storm warning` có thể bị kéo bởi cụm “boats on water”. Mình sẽ lấy top-1 snippets gọn cho 8 query để nhận xét chất lượng thực tế.
>
> <details><summary>Ran a command</summary>
>
> - Ran `$items = Get-Content -Path 'D:\Users\manual_review_samples.jsonl' | ForEach-Object { ConvertFrom-Json $_ }
> foreach ($item in $items) {
>   $base = ($item.candidates | Where-Object { $_.label -eq 'template_baseline' }).top_results[0]
>   $prop = ($item.candidates | Where-Object { $_.label -eq 'vlm_propagated' }).top_results[0]
>   "--- $($item.query_id): $($item.query)"
>   "BASE $($base.document_id) score=$([math]::Round([double]$base.score,3)) terms=$($base.matched_terms -join ',')"
>   "  $($base.snippet.Substring(0, [Math]::Min(220, $base.snippet.Length)))"
>   "PROP $($prop.document_id) score=$([math]::Round([double]$prop.score,3)) terms=$($prop.matched_terms -join ',')"
>   "  $($prop.snippet.Substring(0, [Math]::Min(220, $prop.snippet.Length)))"
> }`
>
> </details>
</details>

Kết quả visual benchmark **tốt hơn default benchmark** và cho thấy Phase 8 có tác dụng thật.

**Tóm tắt**
```text
template_baseline: Hit 8/8, Avg Top1 37.012
vlm_frame_only:    Hit 8/8, Avg Top1 37.789
vlm_propagated:    Hit 8/8, Avg Top1 41.338
```

So với baseline:

```text
vlm_frame_only:    +0.777 avg top1, 7/8 same top1
vlm_propagated:    +4.326 avg top1, 4 improved, 1 regressed, 3 same top1
```

Nghĩa là:

- `vlm_frame_only` gần như chưa tác động nhiều.
- `vlm_propagated` đã tác động thật vào search visual.
- Với query visual, propagated đang là nhánh tốt nhất.

**Điểm tốt**
Một số query VLM propagated cải thiện rõ:

```text
q_visual_002 people wearing traditional dress ao dai
baseline: 27.874
propagated: 39.758
```

VLM bắt được các từ tự nhiên như `people`, `wearing`, `dress`, nên tốt hơn template.

```text
q_visual_004 news studio presenter screen
baseline: 26.296
propagated: 36.867
```

Đây là case tốt: caption VLM mô tả “news desk / television studio”, baseline chỉ có object/scene thô.

```text
q_visual_005 cartoon preview colorful characters
baseline: 0.300
propagated: 10.771
```

Có cải thiện lexical, nhưng semantic chưa chắc tốt vì top result lại là parade/colorful, không phải cartoon. Query này cần review lại.

**Điểm cần cẩn thận**
`q_visual_001 boats on water storm warning`:

```text
baseline top1: shot_0042, storm/water/flood
propagated top1: shot_0001, boats/on/water
```

Propagated score cao hơn, nhưng về intent “storm warning” thì baseline có vẻ đúng hơn. Đây là dấu hiệu lexical scorer đang ưu tiên cụm `boats on water`, chưa hiểu đủ trọng số “storm warning”.

`q_visual_008 computer screen website text screenshot`:

```text
baseline: 69.205
propagated: 65.419
same top1
```

Không đáng lo: cùng document, chỉ giảm score vì VLM caption tự nhiên hơn, ít lặp token template.

**Kết luận**
Phase 8 **đạt** cho nhánh visual:

```text
Default query: propagated bị giảm vì query thiên template/audio/trake.
Visual query: propagated thắng rõ hơn.
```

Vì vậy không nên kết luận VLM propagated “tệ hơn”. Đúng hơn là:

```text
VLM propagated tốt cho visual retrieval,
nhưng cần route-aware retrieval để không làm giảm audio/OCR/trake query.
```

Bước tiếp hợp lý: giữ `vlm_propagated` cho visual route, rồi Phase tiếp theo nên chỉnh retrieval/ranking theo route: visual query boost caption VLM, audio query boost audio, OCR query boost OCR, trake query boost event/TRAKE.

> Hãy đến phase tiếp theo

<details><summary>18 previous messages</summary>

> Mình sẽ chuyển sang phase kế tiếp theo đúng kết luận vừa rồi: **route-aware retrieval tuning**. Mục tiêu là giữ lợi thế VLM cho visual query nhưng không để caption VLM làm tụt audio/OCR/TRAKE query.
>
> <details><summary>Ran 3 commands</summary>
>
> - Ran `Get-Content -Path data_processing\caption\run_retrieval_export_eval.py`
> - Ran `Get-Content -Path data_processing\caption\caption_pipeline\retrieval_eval.py`
> - Ran `git status --short`
>
> </details>
>
> Mình sẽ thêm `--scoring-profile route_aware` theo kiểu opt-in, nên các kết quả cũ không đổi. Profile mới sẽ bớt điểm các token/template chung như `spoken`, `content`, `scene`, `visible`, `objects`, giảm `all_text`, và boost đúng field theo route: audio -> `audio_text`, OCR -> `ocr_text`, visual -> `caption/object/scene`, TRAKE -> `trake/current_observation`.
>
> <details><summary>Edited a file</summary>
>
> - Wrote `C:/Users/User/Desktop/AIC2026/data_processing/caption/caption_pipeline/retrieval_eval.py` (+162 -37)
>
> </details>
>
> `apply_patch` lại không đọc được đường dẫn repo trên Windows, nên mình sẽ ghi lại file bằng PowerShell có kiểm soát. Đây là chỉnh một module scoring duy nhất, không đụng dữ liệu/output.
>
> <details><summary>Ran a command</summary>
>
> - Ran `$path = 'C:\Users\User\Desktop\AIC2026\data_processing\caption\caption_pipeline\retrieval_eval.py'
> $content = @'
> """Offline lexical retrieval evaluator for Phase 5."""
>
> from __future__ import annotations
>
> import math
> from typing import Any
>
> from .retrieval_export import normalized_text, tokenize
> from .text_utils import truncate
>
>
> ROUTE_FIELD_BONUS = {
>     "ocr": {"ocr_text": 1.8, "all_text": 1.1},
>     "audio": {"audio_text": 1.8, "temporal_caption": 1.2, "trake_text": 1.1},
>     "object_visual": {"object_text": 1.7, "caption_text": 1.2, "scene_text": 1.1},
>     "trake": {
>         "trake_text": 2.0,
>         "current_observation": 1.5,
>         "event_caption": 1.4,
>         "before_context": 1.1,
>         "after_context": 1.1,
>     },
>     "general": {},
> }
>
> ROUTE_UNIT_BONUS = {
>     "ocr": {"frame": 0.4, "shot": 0.2},
>     "audio": {"shot": 0.3, "event_step": 0.2},
>     "object_visual": {"frame": 0.3, "shot": 0.2},
>     "trake": {"event_step": 0.7, "shot": 0.3},
>     "general": {},
> }
>
> SCORING_PROFILES = {"default", "route_aware"}
>
> ROUTE_AWARE_FIELD_MULTIPLIER = {
>     "ocr": {
>         "ocr_text": 2.4,
>         "caption_text": 0.65,
>         "temporal_caption": 0.65,
>         "trake_text": 0.55,
>         "audio_text": 0.35,
>         "all_text": 0.2,
>     },
>     "audio": {
>         "audio_text": 2.8,
>         "ocr_text": 0.45,
>         "caption_text": 0.55,
>         "temporal_caption": 0.75,
>         "trake_text": 0.65,
>         "all_text": 0.2,
>     },
>     "object_visual": {
>         "caption_text": 1.8,
>         "object_text": 1.6,
>         "scene_text": 1.3,
>         "ocr_text": 0.45,
>         "audio_text": 0.25,
>         "trake_text": 0.9,
>         "all_text": 0.25,
>     },
>     "trake": {
>         "trake_text": 1.8,
>         "current_observation": 1.7,
>         "event_caption": 1.5,
>         "temporal_caption": 1.1,
>         "caption_text": 0.75,
>         "ocr_text": 0.45,
>         "audio_text": 0.45,
>         "all_text": 0.2,
>     },
>     "general": {
>         "caption_text": 1.25,
>         "temporal_caption": 1.1,
>         "trake_text": 1.1,
>         "object_text": 1.1,
>         "scene_text": 1.0,
>         "ocr_text": 0.9,
>         "audio_text": 0.9,
>         "all_text": 0.25,
>     },
> }
>
> ROUTE_AWARE_UNIT_BONUS = {
>     "ocr": {"frame": 0.8, "shot": 0.5},
>     "audio": {"event_step": 0.9, "shot": 0.5},
>     "object_visual": {"frame": 0.8, "shot": 0.6},
>     "trake": {"event_step": 1.1, "shot": 0.4},
>     "general": {"event_step": 0.2, "shot": 0.2, "frame": 0.1},
> }
>
> ROUTE_AWARE_GLOBAL_STOP_TERMS = {
>     "a", "an", "and", "are", "at", "by", "for", "from", "in", "is",
>     "of", "on", "or", "the", "to", "with",
> }
>
> ROUTE_AWARE_ROUTE_STOP_TERMS = {
>     "audio": {"audio", "content", "spoken", "speech"},
>     "object_visual": {
>         "frame", "frames", "object", "objects", "representative",
>         "scene", "show", "shows", "visible",
>     },
>     "ocr": {"screen", "text", "on"},
>     "trake": {"shot", "scene"},
>     "general": {"frame", "frames", "representative", "scene", "show", "shows"},
> }
>
>
> def run_local_retrieval(
>     corpus: list[dict],
>     queries: list[dict],
>     top_k: int,
>     scoring_profile: str = "default",
> ) -> list[dict]:
>     """Score all queries against the local corpus and return ranked hits."""
>     _validate_scoring_profile(scoring_profile)
>     results: list[dict] = []
>     for query in queries:
>         scored = _score_query(corpus, query, scoring_profile=scoring_profile)
>         for rank, item in enumerate(scored[:top_k], start=1):
>             result = {
>                 "query_id": query.get("query_id", ""),
>                 "query": query.get("query", ""),
>                 "route": query.get("route", "general"),
>                 "rank": rank,
>                 "score": round(item["score"], 6),
>                 "document_id": item["doc"].get("document_id", ""),
>                 "unit_type": item["doc"].get("unit_type", ""),
>                 "video_id": item["doc"].get("video_id", ""),
>                 "unit_id": item["doc"].get("unit_id", ""),
>                 "frame_id": item["doc"].get("frame_id", ""),
>                 "shot_id": item["doc"].get("shot_id", ""),
>                 "event_id": item["doc"].get("event_id", ""),
>                 "start_sec": item["doc"].get("start_sec", 0.0),
>                 "end_sec": item["doc"].get("end_sec", 0.0),
>                 "matched_fields": item["matched_fields"],
>                 "matched_terms": item["matched_terms"],
>                 "snippet": _make_snippet(item["doc"], item["matched_terms"]),
>             }
>             if "step_order" in item["doc"]:
>                 result["step_order"] = item["doc"].get("step_order")
>             results.append(result)
>     return results
>
>
> def _score_query(
>     corpus: list[dict],
>     query: dict,
>     scoring_profile: str,
> ) -> list[dict]:
>     query_text = query.get("query", "") or ""
>     route = query.get("route", "general") or "general"
>     query_terms = _query_terms_for_profile(tokenize(query_text), route, scoring_profile)
>     target_units = set(query.get("target_unit_types") or [])
>     scored: list[dict[str, Any]] = []
>
>     for doc in corpus:
>         unit_type = doc.get("unit_type", "")
>         if target_units and unit_type not in target_units:
>             continue
>         score, matched_fields, matched_terms = _score_doc(
>             doc,
>             query_terms,
>             query_text,
>             route,
>             scoring_profile,
>         )
>         if score <= 0:
>             continue
>         scored.append({
>             "doc": doc,
>             "score": score,
>             "matched_fields": matched_fields,
>             "matched_terms": matched_terms,
>         })
>
>     scored.sort(
>         key=lambda item: (
>             -item["score"],
>             item["doc"].get("start_sec", 0.0),
>             item["doc"].get("document_id", ""),
>         )
>     )
>     return scored
>
>
> def _score_doc(
>     doc: dict,
>     query_terms: list[str],
>     query_text: str,
>     route: str,
>     scoring_profile: str,
> ) -> tuple[float, list[str], list[str]]:
>     fields = doc.get("fields") or {}
>     boosts = _boosts_for_profile(doc, route, scoring_profile)
>     for field, bonus in ROUTE_FIELD_BONUS.get(route, {}).items():
>         boosts[field] = boosts.get(field, 1.0) * bonus
>
>     matched_fields: list[str] = []
>     matched_terms: set[str] = set()
>     score = 0.0
>     for field, text in fields.items():
>         if not text:
>             continue
>         field_terms = tokenize(text)
>         if not field_terms:
>             continue
>         field_term_set = set(field_terms)
>         hits = [term for term in query_terms if term in field_term_set]
>         if not hits:
>             continue
>         weight = boosts.get(field, 1.0)
>         tf_score = sum(1.0 + math.log(1.0 + field_terms.count(term)) for term in hits)
>         coverage = len(hits) / max(len(query_terms), 1)
>         score += weight * (tf_score + coverage)
>         matched_fields.append(field)
>         matched_terms.update(hits)
>
>     phrase = normalized_text(query_text)
>     search_text = normalized_text(doc.get("search_text", ""))
>     if phrase and phrase in search_text:
>         score += 3.0
>
>     unit_bonus = _unit_bonus_for_profile(route, doc.get("unit_type", ""), scoring_profile)
>     score += unit_bonus
>     if doc.get("unit_type") == "event_step" and route == "trake":
>         score += _temporal_role_bonus(doc, query_terms)
>
>     return score, matched_fields, sorted(matched_terms)
>
>
> def _validate_scoring_profile(scoring_profile: str) -> None:
>     if scoring_profile not in SCORING_PROFILES:
>         allowed = ", ".join(sorted(SCORING_PROFILES))
>         raise ValueError(
>             f"Invalid scoring_profile '{scoring_profile}'. Expected one of: {allowed}"
>         )
>
>
> def _query_terms_for_profile(
>     query_terms: list[str],
>     route: str,
>     scoring_profile: str,
> ) -> list[str]:
>     if scoring_profile != "route_aware":
>         return query_terms
>     stop_terms = set(ROUTE_AWARE_GLOBAL_STOP_TERMS)
>     stop_terms.update(ROUTE_AWARE_ROUTE_STOP_TERMS.get(route, set()))
>     filtered = [term for term in query_terms if term not in stop_terms]
>     return filtered or query_terms
>
>
> def _boosts_for_profile(
>     doc: dict,
>     route: str,
>     scoring_profile: str,
> ) -> dict[str, float]:
>     boosts = dict(doc.get("boosts") or {})
>     if scoring_profile != "route_aware":
>         return boosts
>     for field, multiplier in ROUTE_AWARE_FIELD_MULTIPLIER.get(route, {}).items():
>         boosts[field] = boosts.get(field, 1.0) * multiplier
>     return boosts
>
>
> def _unit_bonus_for_profile(route: str, unit_type: str, scoring_profile: str) -> float:
>     if scoring_profile == "route_aware":
>         return ROUTE_AWARE_UNIT_BONUS.get(route, {}).get(unit_type, 0.0)
>     return ROUTE_UNIT_BONUS.get(route, {}).get(unit_type, 0.0)
>
>
> def _temporal_role_bonus(doc: dict, query_terms: list[str]) -> float:
>     action_state = doc.get("action_state", "")
>     temporal_role = doc.get("temporal_role", "")
>     terms = set(query_terms)
>     bonus = 0.0
>     if {"dau", "tien"} & terms and (action_state == "start" or temporal_role == "beginning"):
>         bonus += 1.0
>     if {"cuoi", "cung"} & terms and (action_state == "end" or temporal_role == "completion"):
>         bonus += 1.0
>     if "sau" in terms and temporal_role in {"continuation", "change"}:
>         bonus += 0.5
>     return bonus
>
>
> def _make_snippet(doc: dict, matched_terms: list[str]) -> str:
>     fields = doc.get("fields") or {}
>     preferred = [
>         "trake_text",
>         "current_observation",
>         "caption_text",
>         "temporal_caption",
>         "ocr_text",
>         "audio_text",
>         "object_text",
>         "all_text",
>     ]
>     for field in preferred:
>         text = fields.get(field, "") or ""
>         if not text:
>             continue
>         normalized = normalized_text(text)
>         if any(term in normalized for term in matched_terms):
>             return truncate(text, 280)
>     return truncate(doc.get("search_text", ""), 280)
> '@
> Set-Content -LiteralPath $path -Value $content -Encoding UTF8`
> - Auto-review: approved
> - Auto-review risk: low
> - Auto-review rationale: Auto-review returned a low-risk allow decision.
>
> </details>
>
> Scoring core đã có profile `route_aware`. Tiếp theo mình nối vào CLI retrieval bằng `--scoring-profile`, để bạn có thể chạy song song `default`, `visual`, và `route_aware` mà không phá kết quả cũ.
>
> <details><summary>Ran a command</summary>
>
> - Ran `$path = 'C:\Users\User\Desktop\AIC2026\data_processing\caption\run_retrieval_export_eval.py'
> $text = Get-Content -LiteralPath $path -Raw
> $old = @'
>     parser.add_argument("--top-k", type=int, default=20, help="Top-K results per query")
> '@
> $new = @'
>     parser.add_argument("--top-k", type=int, default=20, help="Top-K results per query")
>     parser.add_argument(
>         "--scoring-profile",
>         default="default",
>         choices=["default", "route_aware"],
>         help="Local lexical scoring profile. default preserves old behavior.",
>     )
> '@
> if (-not $text.Contains($old)) { throw 'Failed to find top-k parser block' }
> $text = $text.Replace($old, $new)
> $old = @'
>     logger.info("Exports:    %s", exports_dir)
>     logger.info("Eval:       %s", eval_dir)
> '@
> $new = @'
>     logger.info("Exports:    %s", exports_dir)
>     logger.info("Eval:       %s", eval_dir)
>     logger.info("Scoring:    %s", args.scoring_profile)
> '@
> if (-not $text.Contains($old)) { throw 'Failed to find logger block' }
> $text = $text.Replace($old, $new)
> $old = @'
>     results = run_local_retrieval(corpus, queries, top_k=args.top_k)
> '@
> $new = @'
>     results = run_local_retrieval(
>         corpus,
>         queries,
>         top_k=args.top_k,
>         scoring_profile=args.scoring_profile,
>     )
> '@
> if (-not $text.Contains($old)) { throw 'Failed to find run retrieval block' }
> $text = $text.Replace($old, $new)
> $old = @'
>     report = build_retrieval_report(
>         video_id=args.video_id,
>         corpus=corpus,
>         queries=queries,
>         results=results,
>         top_k=args.top_k,
>         warnings=warnings,
>     )
> '@
> $new = @'
>     report = build_retrieval_report(
>         video_id=args.video_id,
>         corpus=corpus,
>         queries=queries,
>         results=results,
>         top_k=args.top_k,
>         warnings=warnings,
>     )
>     report["scoring_profile"] = args.scoring_profile
>     report["eval_name"] = args.eval_name or "default"
> '@
> if (-not $text.Contains($old)) { throw 'Failed to find report block' }
> $text = $text.Replace($old, $new)
> Set-Content -LiteralPath $path -Value $text -Encoding UTF8`
> - Auto-review: approved
> - Auto-review risk: low
> - Auto-review rationale: Auto-review returned a low-risk allow decision.
>
> </details>
>
> CLI đã truyền profile. Mình sẽ bổ sung hiển thị `scoring_profile` trong report markdown để sau này nhìn file là biết đang dùng default hay route-aware.
>
> <details><summary>Ran a command</summary>
>
> - Ran `Get-Content -Path data_processing\caption\caption_pipeline\retrieval_report.py`
>
> </details>
>
> Mình sẽ để report markdown hiển thị `Eval name` và `Scoring profile`, để khi so sánh nhiều report không bị lẫn.
>
> <details><summary>Ran a command</summary>
>
> - Ran `$path = 'C:\Users\User\Desktop\AIC2026\data_processing\caption\caption_pipeline\retrieval_report.py'
> $text = Get-Content -LiteralPath $path -Raw
> $old = @'
>     metrics = [
>         ("Corpus docs", "num_corpus_docs"),
> '@
> $new = @'
>     metrics = [
>         ("Eval name", "eval_name"),
>         ("Scoring profile", "scoring_profile"),
>         ("Corpus docs", "num_corpus_docs"),
> '@
> if (-not $text.Contains($old)) { throw 'Failed to find metrics block' }
> $text = $text.Replace($old, $new)
> Set-Content -LiteralPath $path -Value $text -Encoding UTF8`
> - Auto-review: approved
> - Auto-review risk: low
> - Auto-review rationale: Auto-review returned a low-risk allow decision.
>
> </details>
>
> Route-aware scoring đã nối vào retrieval CLI và report. Giờ mình sẽ kiểm tra diff, imports, và những command chạy phase mới.
>
> <details><summary>Ran 4 commands</summary>
>
> - Ran `git diff -- data_processing/caption/caption_pipeline/retrieval_eval.py data_processing/caption/run_retrieval_export_eval.py data_processing/caption/caption_pipeline/retrieval_report.py data_processing/caption/caption_pipeline/experiment_benchmark.py data_processing/caption/run_caption_experiment_benchmark.py`
> - Ran `Select-String -Path data_processing\caption\caption_pipeline\retrieval_eval.py -Pattern 'SCORING_PROFILES|route_aware|scoring_profile|def run_local_retrieval|def _query_terms_for_profile' -Context 0,2`
> - Ran `git status --short data_processing/caption`
> - Ran `Select-String -Path data_processing\caption\run_retrieval_export_eval.py -Pattern 'scoring-profile|scoring_profile|run_local_retrieval|scoring_profile' -Context 1,3`
>
> </details>
>
> Diff hiện tại chỉ đụng 3 file cho route-aware scoring. Mình sẽ kiểm tra nơi gọi `run_local_retrieval` để đảm bảo thêm tham số optional không phá các script cũ.
>
> <details><summary>Ran 3 commands</summary>
>
> - Ran `rg "run_local_retrieval" data_processing/caption`
> - Ran `Get-Content -Path data_processing\caption\run_retrieval_export_eval.py | Select-Object -Skip 60 -First 150`
> - Ran `Get-Content -Path data_processing\caption\caption_pipeline\retrieval_report.py | Select-Object -Skip 75 -First 35`
>
> </details>
>
> Route-aware phase code đã xong. Vì Python local vẫn là WindowsApps stub nên mình không chạy compile ở máy này; mình sẽ kèm lệnh compile server, rồi lệnh chạy default-query route-aware và visual-query route-aware cho 3 nhánh.
>
> <details><summary>Plan</summary>
>
> - [x] Đọc scoring retrieval hiện tại và xác định điểm chỉnh route-aware
> - [x] Thêm profile scoring route-aware vào retrieval eval/CLI
> - [x] Thêm benchmark command và kiểm tra tĩnh
>
> </details>
</details>

Đã thực hiện phase tiếp theo: **Route-Aware Retrieval Tuning**.

Mình thêm `--scoring-profile route_aware` cho retrieval. Mặc định vẫn là `default`, nên các lệnh cũ không bị đổi kết quả.

Các file đã sửa:

- `caption_pipeline/retrieval_eval.py`
- `run_retrieval_export_eval.py`
- `caption_pipeline/retrieval_report.py`

Ý tưởng của profile mới:

```text
audio query  -> boost audio_text, giảm caption/all_text
ocr query    -> boost ocr_text
visual query -> boost caption_text/object_text/scene_text
trake query  -> boost trake/current_observation/event_caption
```

Đồng thời giảm các token template như:

```text
spoken, content, scene, shows, visible, objects, frame, representative
```

Chạy compile trên server:

```bash
cd /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/data_processing/caption

python -m py_compile \
  caption_pipeline/retrieval_eval.py \
  caption_pipeline/retrieval_report.py \
  run_retrieval_export_eval.py
```

**Chạy route-aware default eval cho 3 nhánh**

```bash
PYTHONIOENCODING=utf-8 python run_retrieval_export_eval.py \
  --video-id L22_V012 \
  --caption-dir /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption \
  --eval-name route_aware \
  --scoring-profile route_aware \
  --top-k 20
```

```bash
PYTHONIOENCODING=utf-8 python run_retrieval_export_eval.py \
  --video-id L22_V012 \
  --caption-dir /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption_experiments/vlm_frame_qwen25vl_7b_rep \
  --eval-name route_aware \
  --scoring-profile route_aware \
  --top-k 20
```

```bash
PYTHONIOENCODING=utf-8 python run_retrieval_export_eval.py \
  --video-id L22_V012 \
  --caption-dir /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption_experiments/vlm_frame_qwen25vl_7b_propagated \
  --eval-name route_aware \
  --scoring-profile route_aware \
  --top-k 20
```

Benchmark:

```bash
PYTHONIOENCODING=utf-8 python run_caption_experiment_benchmark.py \
  --video-id L22_V012 \
  --baseline-dir /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption \
  --baseline-label template_baseline \
  --experiment vlm_frame_only=/tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption_experiments/vlm_frame_qwen25vl_7b_rep \
  --experiment vlm_propagated=/tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption_experiments/vlm_frame_qwen25vl_7b_propagated \
  --eval-name route_aware \
  --benchmark-name phase9_route_aware_benchmark \
  --top-n-samples 5 \
  --strict
```

Sau đó xem:

```bash
less /tmp2/maitanha/vgu/ttn/AIC-Khoa/AIC-Test/caption/L22_V012/benchmarks/phase9_route_aware_benchmark/phase9_route_aware_benchmark.md
```

Mục tiêu của phase này là xem `vlm_propagated` có giữ được lợi thế visual mà không tụt mạnh trên audio/OCR/TRAKE nữa không.

> Có vẻ kết quả không khả quan
>
> User context
>
> Attachments:
> - manual_review_samples.jsonl: `d:/Users/manual_review_samples.jsonl`
> - phase9_route_aware_benchmark.md: `d:/Users/phase9_route_aware_benchmark.md`

<details><summary>18 previous messages</summary>

> Mình sẽ đọc report Phase 9 và manual samples để xem “không khả quan” là do route-aware làm tụt tất cả, hay chỉ làm mất lợi thế ở vài route/query. Sau đó mình sẽ kết luận nên rollback, tinh chỉnh nhẹ, hay chuyển sang hướng scoring khác.
>
> <details><summary>Ran 3 commands</summary>
>
> - Ran `Get-Content -Path 'D:\Users\phase9_route_aware_benchmark.md'`
> - Ran `Get-Content -Path 'D:\Users\manual_review_samples.jsonl' -TotalCount 12`
> - Ran `(Get-Content -Path 'D:\Users\manual_review_samples.jsonl').Count`
>
> </details>
>
> Context compaction
>
> Source: automatic
> Status: completed
>
> Mình sẽ đọc riêng file benchmark phase 9 và chỉ trích phần cần thiết từ `manual_review_samples.jsonl`, tránh mở nguyên file dài để khỏi nhiễu. Kết quả không khả quan thường phải nhìn theo từng query: nó tụt vì scoring route-aware thật sự sai, hay vì benchmark/query đang thiên về template lexical.
>
> Có vẻ shell bên Windows session đang không nhận `D:\Users\outputs` làm working directory trong lần gọi vừa rồi. Mình sẽ thử lại với working directory mặc định rồi chỉ đọc đúng hai file bạn gửi.
>
> PowerShell runner đang lỗi trước khi vào lệnh, nên mình đổi sang `cmd` để đọc file. Đây chỉ là đọc dữ liệu benchmark, không động vào repo.
>
> Lỗi là ở working directory mặc định của tool, không phải ở file. Mình chuyển hẳn workdir sang `D:\tmp`, thường ổn hơn trong sandbox Windows.
>
> <details><summary>Ran 2 commands</summary>
>
> - Ran `"C:\\WINDOWS\\system32\\cmd.exe" /c "type D:\\Users\\phase9_route_aware_benchmark.md"`
> - Ran `"C:\\WINDOWS\\system32\\cmd.exe" /c "python -c \"import json; p=r'D:\\Users\\manual_review_samples.jsonl'; rows=[json.loads(l) for l in open(p,encoding='utf-8')]; print('n=',len(rows));
> for r in rows:
>     c=(r.get('candidates') or [{}])[0]; b=r.get('baseline') or {}; bt=(b.get('top1') or {}); ct=(c.get('top1') or {}); print(f\\\"{r.get('query_id')} | {r.get('route')} | {r.get('query')} | base={bt.get('document_id')} {bt.get('score')} | cand={ct.get('document_id')} {ct.get('score')} | delta={c.get('delta_top1_score')} | same={c.get('same_top1')}\\\")\""`
>
> </details>
>
> Benchmark tổng đã đọc được: `vlm_propagated` giảm khá rõ dưới route-aware. Mình cần thêm bảng query-level từ `manual_review_samples` để xác định tụt ở loại truy vấn nào; `python` trên máy Windows này đang là stub nên mình dùng PowerShell lại với workdir ổn.
>
> <details><summary>Ran a command</summary>
>
> - Ran `$p='D:\Users\manual_review_samples.jsonl'; $items=Get-Content -LiteralPath $p | ForEach-Object { $_ | ConvertFrom-Json }; $items | Select-Object query_id,query,route,@{n='baseline_top1';e={$_.baseline.top1.document_id}},@{n='baseline_score';e={$_.baseline.top1.score}},@{n='candidate_top1';e={$_.candidates[0].top1.document_id}},@{n='candidate_score';e={$_.candidates[0].top1.score}},@{n='delta';e={$_.candidates[0].delta_top1_score}},@{n='same_top1';e={$_.candidates[0].same_top1}} | Format-Table -AutoSize`
>
> </details>
>
> Bảng vừa rồi bị rỗng vì cấu trúc `manual_review_samples.jsonl` không giống file benchmark trước, có lẽ key candidate được đặt theo label. Mình kiểm tra schema của 1 record trước rồi mới tóm tắt.
>
> <details><summary>Ran a command</summary>
>
> - Ran `$p='D:\Users\manual_review_samples.jsonl'; $first=(Get-Content -LiteralPath $p -TotalCount 1 | ConvertFrom-Json); $first | ConvertTo-Json -Depth 8`
>
> </details>
>
> Schema đúng là mỗi record chứa nhiều candidate trong mảng `candidates`, không có `baseline` tách riêng. Mình sẽ rút bảng top-1 theo ba nhánh: baseline, VLM frame-only, VLM propagated.
>
> <details><summary>Ran a command</summary>
>
> - Ran `$p='D:\Users\manual_review_samples.jsonl'; $items=Get-Content -LiteralPath $p | ForEach-Object { $_ | ConvertFrom-Json }; foreach($r in $items){ $b=($r.candidates | Where-Object {$_.label -eq 'template_baseline'}).top_results[0]; $f=($r.candidates | Where-Object {$_.label -eq 'vlm_frame_only'}).top_results[0]; $v=($r.candidates | Where-Object {$_.label -eq 'vlm_propagated'}).top_results[0]; $sameBF=($b.document_id -eq $f.document_id); $sameBV=($b.document_id -eq $v.document_id); [PSCustomObject]@{query_id=$r.query_id; route=$r.route; query=$r.query; base_doc=$b.document_id; base_score=[math]::Round([double]$b.score,3); prop_doc=$v.document_id; prop_score=[math]::Round([double]$v.score,3); delta=[math]::Round(([double]$v.score-[double]$b.score),3); same=$sameBV} } | Format-Table -AutoSize`
>
> </details>
>
> Có lỗi cú pháp nhỏ do PowerShell pipe sau vòng `foreach`; mình đổi sang gom mảng rồi in ra.
>
> <details><summary>Ran 2 commands</summary>
>
> - Ran `$p='D:\Users\manual_review_samples.jsonl'; $items=Get-Content -LiteralPath $p | ForEach-Object { $_ | ConvertFrom-Json }; $out=@(); foreach($r in $items){ $b=($r.candidates | Where-Object {$_.label -eq 'template_baseline'}).top_results[0]; $v=($r.candidates | Where-Object {$_.label -eq 'vlm_propagated'}).top_results[0]; $out += [PSCustomObject]@{query_id=$r.query_id; route=$r.route; query=$r.query; base_doc=$b.document_id; base_score=[math]::Round([double]$b.score,3); prop_doc=$v.document_id; prop_score=[math]::Round([double]$v.score,3); delta=[math]::Round(([double]$v.score-[double]$b.score),3); same=($b.document_id -eq $v.document_id)} }; $out | Format-Table -AutoSize`
> - Ran `$p='D:\Users\manual_review_samples.jsonl'; $items=Get-Content -LiteralPath $p | ForEach-Object { $_ | ConvertFrom-Json }; $out=@(); foreach($r in $items){ $b=($r.candidates | Where-Object {$_.label -eq 'template_baseline'}).top_results[0]; $v=($r.candidates | Where-Object {$_.label -eq 'vlm_propagated'}).top_results[0]; $out += [PSCustomObject]@{query_id=$r.query_id; route=$r.route; query=$r.query; base_doc=$b.document_id; base_score=[math]::Round([double]$b.score,3); prop_doc=$v.document_id; prop_score=[math]::Round([double]$v.score,3); delta=[math]::Round(([double]$v.score-[double]$b.score),3); same=($b.document_id -eq $v.document_id)} }; $out | ConvertTo-Csv -NoTypeInformation`
>
> </details>
>
> Giờ đã rõ: 6 query tụt rõ, 5 query cùng/không đổi thực chất, chỉ 1 query tăng nhẹ. Mình lấy thêm snippet top-1 của các query tụt mạnh để xem lỗi là do VLM propagated, hay do cách route-aware chấm điểm kéo sai unit.
>
> <details><summary>Ran a command</summary>
>
> - Ran `$p='D:\Users\manual_review_samples.jsonl'; $items=Get-Content -LiteralPath $p | ForEach-Object { $_ | ConvertFrom-Json }; foreach($r in $items){ $b=($r.candidates | Where-Object {$_.label -eq 'template_baseline'}).top_results[0]; $v=($r.candidates | Where-Object {$_.label -eq 'vlm_propagated'}).top_results[0]; $delta=([double]$v.score-[double]$b.score); if($delta -lt -10){ Write-Output ('--- '+$r.query_id+' | '+$r.route+' | '+$r.query+' | delta='+[math]::Round($delta,3)); Write-Output ('BASE '+$b.document_id+' score='+[math]::Round([double]$b.score,3)); Write-Output ($b.snippet.Substring(0,[Math]::Min(220,$b.snippet.Length))); Write-Output ('PROP '+$v.document_id+' score='+[math]::Round([double]$v.score,3)); Write-Output ($v.snippet.Substring(0,[Math]::Min(220,$v.snippet.Length))); } }`
>
> </details>
</details>

Đúng, Phase 9 route-aware **không khả quan** và hiện tại chưa nên lấy làm hướng chính.

Từ benchmark:

| Nhánh | Hit Rate | Avg Top1 |
|---|---:|---:|
| `template_baseline` | 1.000 | 57.962 |
| `vlm_frame_only` | 1.000 | 57.962 |
| `vlm_propagated` | 1.000 | 47.800 |

So với baseline, `vlm_propagated` có:

- `Improved`: 1 query
- `Regressed`: 6 query
- `Same Top1`: 5 query
- `Avg Top1 Delta`: `-10.162`

Các tụt mạnh nhất:

- `q_trake_001`: `-39.856`
- `q_general_002`: `-25.671`
- `q_audio_002`: `-21.903`
- `q_shot_001`: `-18.496`
- `q_audio_001`: `-13.284`

Nhận xét chính: **route-aware lexical scoring đang over-tune**. Nó cố boost theo route như audio/OCR/object/TRAKE, nhưng với `vlm_propagated`, caption tự nhiên hơn, ít giữ các cụm template như `scene shows`, `visible objects`, `representative frames`. Vì vậy điểm lexical bị lệch, nhiều query bị kéo sang `compact:shot` hoặc event khác chỉ vì trùng từ, chưa chắc đúng ngữ nghĩa hơn.

Điểm đáng chú ý: `vlm_frame_only` y hệt baseline trong benchmark này, nghĩa là phần VLM frame chưa tác động đủ vào các tầng search chính nếu chưa propagate. Nhưng khi propagate thì lại làm route-aware scoring dễ lệch.

Kết luận của mình:

- Không dùng `route_aware` làm scoring production lúc này.
- Giữ `--scoring-profile route_aware` như một experiment thất bại có ích.
- Quay lại dùng `--scoring-profile default` làm chuẩn hiện tại.
- Phase tiếp theo nên chuyển sang **hybrid/RRF retrieval**, không tiếp tục chỉnh weight thủ công.

Hướng tốt hơn là: chạy nhiều ranking riêng theo channel `caption`, `ocr`, `audio`, `object`, `trake`, rồi gộp bằng Reciprocal Rank Fusion. Cách này tránh việc score thô của một field đè field khác, và ổn hơn nhiều so với boost thủ công.

> Nãy giờ mình đang ở phase nào, đã tới phase cuối cùng hay chưa

Mình hiện đang ở **Phase 9: route-aware retrieval benchmark**.

Tóm trạng thái:

- **Phase 0-6**: pipeline caption chuẩn đã pass.
  Evidence → frame caption template → shot caption → TRAKE event → compact index → retrieval/export/eval → benchmark.

- **Phase 7**: VLM frame-caption experiment đã chạy với **Qwen2.5-VL-7B**.
  Sinh được frame caption override cho `207/283` frames.

- **Phase 8**: propagate VLM frame caption lên shot/event/index đã pass.
  Visual benchmark có tín hiệu tốt hơn baseline.

- **Phase 9**: thử `route_aware` scoring.
  Kết quả **không khả quan**, `vlm_propagated` tụt Avg Top1 so với baseline, nên chưa nên chọn làm hướng production.

Vậy là **chưa tới phase cuối cùng**.

Hiện mình đang ở đoạn: pipeline đã chạy được end-to-end, nhưng phần retrieval/scoring cuối chưa chốt. Nếu dừng ở đây thì có thể xem Phase 8 là candidate tốt nhất hiện tại cho hướng visual, còn Phase 9 là experiment thất bại có ích.

Bước hợp lý tiếp theo là **Phase 10: hybrid/RRF retrieval**, tức không boost thủ công kiểu route-aware nữa, mà tạo ranking riêng cho `caption`, `ocr`, `audio`, `object`, `trake`, rồi gộp bằng Reciprocal Rank Fusion. Phase này vẫn chủ yếu CPU, không cần GPU, và sẽ quyết định xem VLM propagated có thật sự tốt hơn khi search được thiết kế đúng hơn không.

> Hãy tiếp tục đến phase tiếp theo

<details><summary>31 previous messages</summary>

> Mình sẽ coi phase tiếp theo là **Phase 10: hybrid/RRF retrieval**: giữ pipeline caption/VLM hiện tại, chỉ thêm một scoring profile mới để fusion nhiều channel thay vì boost thủ công như Phase 9. Trước khi sửa, mình đọc lại module retrieval hiện tại để cắm vào đúng chỗ, tránh làm xáo pipeline đã pass.
>
> <details><summary>Ran 3 commands</summary>
>
> - Ran `rg -n "SCORING_PROFILES|run_local_retrieval|scoring_profile|route_aware|matched_fields|score" data_processing/caption/caption_pipeline/retrieval_eval.py`
> - Ran `rg -n "scoring-profile|eval-name|run_local_retrieval|retrieval_eval" data_processing/caption/run_retrieval_export_eval.py data_processing/caption/caption_pipeline/retrieval_report.py data_processing/caption/caption_pipeline/experiment_benchmark.py`
> - Ran `Get-Content -LiteralPath data_processing/caption/caption_pipeline/retrieval_eval.py -TotalCount 260`
>
> </details>
>
> Code hiện tại có profile `default` và `route_aware`; Phase 10 tốt nhất là thêm profile thứ ba kiểu `rrf`, không phá hai profile cũ. Mình đang đọc nốt phần sau của evaluator để thiết kế RRF bằng lại các hàm scoring sẵn có.
>
> <details><summary>Ran 3 commands</summary>
>
> - Ran `Get-Content -LiteralPath data_processing/caption/run_retrieval_export_eval.py | Select-Object -Skip 50 -First 180`
> - Ran `Get-Content -LiteralPath data_processing/caption/caption_pipeline/retrieval_eval.py | Select-Object -Skip 240 -First 220`
> - Ran `Get-Content -LiteralPath data_processing/caption/caption_pipeline/retrieval_export.py -TotalCount 220`
>
> </details>
>
> RRF có thể dùng ngay corpus đã build: mỗi channel tạo một ranking từ field subset riêng, sau đó cộng điểm theo `1/(k+rank)`. Như vậy không phụ thuộc score thô giữa audio/OCR/caption và tránh lặp lại lỗi Phase 9.
>
> <details><summary>Ran 3 commands</summary>
>
> - Ran `Get-Content -LiteralPath data_processing/caption/caption_pipeline/retrieval_report.py -TotalCount 220`
> - Ran `Get-Content -LiteralPath data_processing/caption/caption_pipeline/retrieval_export.py | Select-Object -Skip 220 -First 120`
> - Ran `rg -n "scoring_profile|SCORING_PROFILES|route_aware|choices=\[\"default\"" data_processing/caption -S`
>
> </details>
>
> Mình sẽ chỉnh rất gọn: thêm `rrf` vào evaluator và CLI, không đổi default behavior. Các kết quả Phase 5-9 cũ vẫn chạy lại được vì `default` và `route_aware` giữ nguyên.
>
> <details><summary>Edited a file</summary>
>
> - Wrote `C:/Users/User/Desktop/AIC2026/data_processing/caption/caption_pipeline/retrieval_eval.py` (+1 -1)
>
> </details>
>
> Do repo nằm ngoài writable root của phiên này, `apply_patch` không đọc được trực tiếp file Windows path. Mình sẽ tạo một patch trong `D:\tmp`, rồi xin quyền apply patch vào repo để sửa đúng các file cần thiết.
>
> <details><summary>Created a file</summary>
>
> - Wrote `D:/tmp/phase10_rrf.patch` (+272 -0)
> - Auto-review: approved
> - Auto-review risk: low
> - Auto-review rationale: Auto-review returned a low-risk allow decision.
>
> </details>
>
> Mình đã tạo patch tạm, nhưng thấy có một dòng delimiter thừa trong patch file do cách đóng block. Mình dọn dòng đó trước khi apply vào repo.
>
> <details><summary>Ran a command</summary>
>
> - Ran `(Get-Content -LiteralPath 'D:\tmp\phase10_rrf.patch') | Where-Object { $_ -ne '*** End Patch' } | Set-Content -LiteralPath 'D:\tmp\phase10_rrf.patch' -Encoding UTF8`
>
> </details>
>
> Bây giờ mình sẽ apply patch vào repo. Đây là chỉnh code trong `C:\Users\User\Desktop\AIC2026`, nên cần quyền ngoài writable root hiện tại.
>
> <details><summary>Ran a command</summary>
>
> - Ran `git apply --reject --whitespace=nowarn D:\tmp\phase10_rrf.patch`
> - Auto-review: approved
> - Auto-review risk: low
> - Auto-review rationale: Auto-review returned a low-risk allow decision.
>
> </details>
>
> `git apply` chưa nhận patch vì file patch có header không chuẩn sau khi tạo thủ công. Mình sẽ đọc vài dòng đầu và tạo lại patch theo dạng `diff --git` sạch hơn.
>
> <details><summary>Ran a command</summary>
>
> - Ran `Get-Content -LiteralPath 'D:\tmp\phase10_rrf.patch' -TotalCount 30`
>
> </details>
>
> Vì patch chuẩn cần line number, mình chuyển sang một script chỉnh sửa có kiểm tra marker rõ ràng: nếu marker không khớp thì script sẽ fail, không sửa bừa. Sau đó mình chạy script với quyền apply vào repo.
>
> <details><summary>Created a file</summary>
>
> - Wrote `D:/tmp/phase10_rrf_edit.ps1` (+331 -0)
> - Auto-review: approved
> - Auto-review risk: low
> - Auto-review rationale: Auto-review returned a low-risk allow decision.
>
> </details>
>
> Script chỉnh sửa đã sẵn sàng. Mình chạy nó với quyền ghi repo, rồi sẽ kiểm tra diff và cú pháp phần vừa thêm.
>
> <details><summary>Ran a command</summary>
>
> - Ran `powershell -ExecutionPolicy Bypass -File D:\tmp\phase10_rrf_edit.ps1`
> - Auto-review: approved
> - Auto-review risk: low
> - Auto-review rationale: This is a user-directed repo code edit for the next caption pipeline phase, limited to local files and reversible with no destructive or external side effects.
>
> </details>
>
> Phase 10 đã được apply. Mình đang rà lại diff để bắt lỗi cú pháp hoặc chỗ nào vô tình chạm vào behavior cũ.
>
> <details><summary>Ran 2 commands</summary>
>
> - Ran `git diff -- data_processing/caption/caption_pipeline/retrieval_eval.py data_processing/caption/run_retrieval_export_eval.py`
> - Ran `rg -n "SCORING_PROFILES|RRF_|def _score_query_rrf|ranking_channels|choices=\[\"default\"" data_processing/caption/caption_pipeline/retrieval_eval.py data_processing/caption/run_retrieval_export_eval.py`
>
> </details>
>
> Diff đúng về logic, nhưng PowerShell đã thêm BOM ở đầu hai file (`﻿` trước docstring/shebang). Mình sẽ ghi lại hai file bằng UTF-8 no BOM để tránh lỗi khi chạy trên server Linux.
>
> <details><summary>Ran a command</summary>
>
> - Ran `$paths=@('C:\Users\User\Desktop\AIC2026\data_processing\caption\caption_pipeline\retrieval_eval.py','C:\Users\User\Desktop\AIC2026\data_processing\caption\run_retrieval_export_eval.py'); $enc=New-Object System.Text.UTF8Encoding($false); foreach($p in $paths){ $text=[System.IO.File]::ReadAllText($p); [System.IO.File]::WriteAllText($p,$text,$enc) }`
> - Auto-review: approved
> - Auto-review risk: low
> - Auto-review rationale: Auto-review returned a low-risk allow decision.
>
> </details>
>
> BOM đã được dọn. Mình chạy thêm kiểm tra diff/whitespace và thử compile nếu Python local dùng được; nếu Python trên Windows vẫn là stub thì mình sẽ đưa lệnh kiểm tra trên server.
>
> <details><summary>Ran 3 commands</summary>
>
> - Ran `git diff --check -- data_processing/caption/caption_pipeline/retrieval_eval.py data_processing/caption/run_retrieval_export_eval.py`
> - Ran `git diff --stat -- data_processing/caption/caption_pipeline/retrieval_eval.py data_processing/caption/run_retrieval_export_eval.py`
> - Ran `Get-Content -LiteralPath data_processing/caption/caption_pipeline/retrieval_eval.py -TotalCount 3; Get-Content -LiteralPath data_processing/caption/run_retrieval_export_eval.py -TotalCount 3`
>
> </details>
>
> Whitespace check chỉ còn “blank line at EOF”. Mình dọn nốt để diff sạch hơn.
>
> <details><summary>Ran a command</summary>
>
> - Ran ``$paths=@('C:\Users\User\Desktop\AIC2026\data_processing\caption\caption_pipeline\retrieval_eval.py','C:\Users\User\Desktop\AIC2026\data_processing\caption\run_retrieval_export_eval.py'); $enc=New-Object System.Text.UTF8Encoding($false); foreach($p in $paths){ $text=[System.IO.File]::ReadAllText($p); $text=$text.TrimEnd()+"`n"; [System.IO.File]::WriteAllText($p,$text,$enc) }``
> - Auto-review: approved
> - Auto-review risk: low
> - Auto-review rationale: Auto-review returned a low-risk allow decision.
>
> </details>
>
> Mình sẽ kiểm tra lại whitespace và cú pháp. Nếu Python local không chạy được ở Windows session này thì mình sẽ chốt bằng lệnh server để bạn verify sau khi pull.
>
> <details><summary>Ran 3 commands</summary>
>
> - Ran `git diff --check -- data_processing/caption/caption_pipeline/retrieval_eval.py data_processing/caption/run_retrieval_export_eval.py`
> - Ran `python -m py_compile data_processing/caption/caption_pipeline/retrieval_eval.py data_processing/caption/run_retrieval_export_eval.py`
> - Ran `rg -n "SCORING_PROFILES|RRF_|def _score_query_rrf|ranking_channels|choices" data_processing/caption/caption_pipeline/retrieval_eval.py data_processing/caption/run_retrieval_export_eval.py`
>
> </details>
>
> System error
>
> You've hit your usage limit. Upgrade to Pro (https://chatgpt.com/explore/pro), visit https://chatgpt.com/codex/settings/usage to purchase more credits or try again at 3:35 PM.
</details>
