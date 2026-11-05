import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import { App } from "../App";
import type { ConfigDict, ParsedConfig } from "../api/types";
import { createVideoApi, type VideoRecord } from "../api/video";
import { createEditorStore, type EditorStore } from "../editor/yamlState";
import { fakeClient, fakeVideoApi, mcapCatalog, mcapRecord, sampleConfig, videoRecord, type FakeVideoApi } from "../test/fixtures";
import { classifyFiles } from "./DropZone";

async function renderStudio(api: FakeVideoApi = fakeVideoApi(), parsed: ParsedConfig = sampleConfig) {
  const client = fakeClient(mcapCatalog(), parsed);
  const store = createEditorStore(client, { debounceMs: 0 });
  await store.load("ignored", "p.yaml");
  const view = render(<App client={client} store={store} videoApi={api} />);
  await screen.findByRole("list", { name: "Pipeline stages" });
  return { api, store, view };
}

const config = (store: EditorStore) => store.getState().pipeline.config;
const section = (store: EditorStore, key: string) => config(store)[key] as ConfigDict | undefined;
const deck = () => screen.getByRole("region", { name: "Video and processing" });
const topicRadio = (name: string) => screen.getByRole("radio", { name: new RegExp(name.replace(/[/]/g, "\\/")) });

async function openLibrary(name: RegExp) {
  await userEvent.click(await screen.findByRole("button", { name }));
}

function drop(container: HTMLElement, ...files: File[]) {
  const app = container.querySelector(".app") as HTMLElement;
  fireEvent.drop(app, { dataTransfer: { files, types: ["Files"] } });
}

const withCamera = (camera: ConfigDict): ParsedConfig => ({ config: { ...sampleConfig.config, camera }, extras: {} });

describe("MCAP input", () => {
  it("drop accepts .mcap and videos", async () => {
    const f = (name: string, type = "") => new File(["x"], name, { type });
    expect(classifyFiles([f("drive.mcap")]).video?.name).toBe("drive.mcap");
    expect(classifyFiles([f("DRIVE.MCAP")]).video?.name).toBe("DRIVE.MCAP");
    expect(classifyFiles([f("road.mkv")]).video?.name).toBe("road.mkv");

    const { api, view } = await renderStudio();
    expect(screen.getByTestId("video-input").getAttribute("accept")).toContain(".mcap");
    drop(view.container, f("drive.mcap"));
    await waitFor(() => expect(api.uploaded.map((u) => u.name)).toEqual(["drive.mcap"]));
    expect(await within(deck()).findByRole("heading", { name: /drive\.mcap/ })).toBeInTheDocument();
    drop(view.container, f("road.mkv", "video/x-matroska"));
    await waitFor(() => expect(api.uploaded.map((u) => u.name)).toEqual(["drive.mcap", "road.mkv"]));
    expect(await within(deck()).findByLabelText("Video road.mkv")).toBeInTheDocument();
  });

  it("after an MCAP drop the topic list shows name, resolution, rate and CameraInfo badge", async () => {
    const { view } = await renderStudio();
    drop(view.container, new File(["x"], "drive.mcap"));
    const group = await within(deck()).findByRole("group", { name: "Image topics" });
    expect(within(group).getAllByRole("radio")).toHaveLength(4);

    const cam = topicRadio("/cam/image_raw");
    expect(cam).toBeChecked(); // the first supported topic is preselected
    const camRow = cam.closest("li") as HTMLElement;
    expect(camRow).toHaveTextContent("16×12");
    expect(camRow).toHaveTextContent("2 Hz");
    expect(within(camRow).getByText("CameraInfo")).toBeInTheDocument();

    const front = topicRadio("/front/image_raw/compressed").closest("li") as HTMLElement;
    expect(front).toHaveTextContent("12×8");
    expect(front).toHaveTextContent("4 Hz");
    expect(within(front).getByText("no CameraInfo")).toBeInTheDocument();

    expect(within(topicRadio("/rect/image_raw").closest("li") as HTMLElement).getByText("CameraInfo · rectified")).toBeInTheDocument();

    // the unsupported topic is listed first (sorted by name), still with its
    // CameraInfo badge, but disabled with its reason; /cam/image_raw was preselected
    const depth = topicRadio("/cam/depth/image_raw");
    expect(within(group).getAllByRole("radio")[0]).toBe(depth);
    expect(depth).toBeDisabled();
    expect(depth).not.toBeChecked();
    expect(depth.closest("li")).toHaveTextContent("unsupported image encoding '16UC1'");
    expect(within(depth.closest("li") as HTMLElement).getByText("CameraInfo")).toBeInTheDocument();
    await userEvent.click(depth);
    expect(depth).not.toBeChecked();
    expect(cam).toBeChecked();
    expect(within(screen.getByRole("region", { name: "Rectification" })).getByRole("switch", { name: "Rectify the images" })).toBeEnabled(); // still /cam/image_raw's

    // the still image stands in for the player: there is no preview video of an MCAP
    expect(within(deck()).queryByLabelText(/^Video /)).not.toBeInTheDocument();
    expect(within(deck()).getByRole("img", { name: /drive\.mcap/ })).toHaveAttribute("src", mcapRecord.urls.thumb);
  });

  it("selecting a topic writes source.mcap/source.topic into the config", async () => {
    const { api, store } = await renderStudio();
    await openLibrary(/drive\.mcap/);
    await waitFor(() => expect(section(store, "source")).toEqual({ mcap: mcapRecord.original, topic: "/cam/image_raw", stride: 1 }));

    await userEvent.click(topicRadio("/rect/image_raw"));
    expect(section(store, "source")).toEqual({ mcap: mcapRecord.original, topic: "/rect/image_raw", stride: 1 });

    const process = within(deck()).getByRole("button", { name: /^Process$/ });
    await waitFor(() => expect(process).toBeEnabled());
    await userEvent.click(process);
    await waitFor(() => expect(api.submitted).toHaveLength(1));
    expect(api.submitted[0].videoId).toBe(mcapRecord.id);
    const sent = api.submitted[0].config.source as ConfigDict;
    expect(sent).toMatchObject({ mcap: mcapRecord.original, topic: "/rect/image_raw", start: 0, stride: 1 });
    expect(sent).not.toHaveProperty("video");

    // back to a video: the MCAP keys leave the config
    await userEvent.click(within(deck()).getByRole("button", { name: /Change input/ }));
    await openLibrary(/road\.mkv/);
    await waitFor(() => expect(section(store, "source")).toEqual({ stride: 1 }));
    await waitFor(() => expect(within(deck()).getByRole("button", { name: /^Process$/ })).toBeEnabled());
    await userEvent.click(within(deck()).getByRole("button", { name: /^Process$/ }));
    await waitFor(() => expect(api.submitted).toHaveLength(2));
    expect(api.submitted[1].config.source).toEqual({ video: videoRecord.original, stride: 1, start: 0, duration: 3 });
  });

  it('ticking "use CameraInfo" sets camera.calibration.from_source', async () => {
    const { store } = await renderStudio(fakeVideoApi(), withCamera({ focal_px: 1000 }));
    await openLibrary(/drive\.mcap/);
    const tick = await within(deck()).findByRole("checkbox", { name: /Use the CameraInfo/ });
    expect(tick).toBeEnabled();
    expect(tick).not.toBeChecked();
    await userEvent.click(tick);
    expect(tick).toBeChecked();
    // the CameraInfo replaces the other intrinsics sources
    expect(section(store, "camera")).toEqual({ calibration: { from_source: true } });

    // a topic without CameraInfo: the choice is reset and offered no more
    await userEvent.click(topicRadio("/front/image_raw/compressed"));
    expect(section(store, "camera")).toEqual({});
    expect(within(deck()).getByRole("checkbox", { name: /Use the CameraInfo/ })).toBeDisabled();
    expect(deck()).toHaveTextContent("no CameraInfo for this topic");

    await userEvent.click(topicRadio("/rect/image_raw"));
    await userEvent.click(within(deck()).getByRole("checkbox", { name: /Use the CameraInfo/ }));
    expect(section(store, "camera")).toEqual({ calibration: { from_source: true } });
    await userEvent.click(within(deck()).getByRole("checkbox", { name: /Use the CameraInfo/ }));
    expect(section(store, "camera")).toEqual({});
  });

  it("rectification toggle is hidden for a video source", async () => {
    await renderStudio();
    await openLibrary(/road\.mkv/);
    await screen.findByLabelText("Video road.mkv");
    expect(screen.getByRole("region", { name: "Detectors" })).toBeInTheDocument();
    expect(screen.queryByRole("region", { name: "Rectification" })).not.toBeInTheDocument();
    expect(screen.queryByRole("switch", { name: "Rectify the images" })).not.toBeInTheDocument();
  });

  it('rectification toggle is disabled with "no CameraInfo for this topic" when the topic has none', async () => {
    const { store } = await renderStudio();
    await openLibrary(/drive\.mcap/);
    await userEvent.click(topicRadio("/front/image_raw/compressed"));
    const lane = screen.getByRole("region", { name: "Rectification" });
    expect(within(lane).getByRole("switch", { name: "Rectify the images" })).toBeDisabled();
    expect(lane).toHaveTextContent("no CameraInfo for this topic");
    expect(section(store, "rectify")).toBeUndefined();
  });

  it('rectification toggle is disabled with "already rectified" when the CameraInfo has no distortion', async () => {
    await renderStudio();
    await openLibrary(/drive\.mcap/);
    await userEvent.click(topicRadio("/rect/image_raw"));
    const lane = screen.getByRole("region", { name: "Rectification" });
    expect(within(lane).getByRole("switch", { name: "Rectify the images" })).toBeDisabled();
    expect(lane).toHaveTextContent("already rectified");
  });

  it("rectification enabled sets rectify.enabled", async () => {
    const { store } = await renderStudio(fakeVideoApi(), withCamera({ focal_px: 1000 }));
    await openLibrary(/drive\.mcap/);
    const lane = await screen.findByRole("region", { name: "Rectification" });
    const toggle = within(lane).getByRole("switch", { name: "Rectify the images" });
    expect(toggle).toBeEnabled();
    expect(toggle).toHaveAttribute("aria-checked", "false");
    await userEvent.click(toggle);
    expect(section(store, "rectify")).toEqual({ enabled: true, alpha: 0 });
    expect(toggle).toHaveAttribute("aria-checked", "true");
    // rectify takes K from the CameraInfo: a fixed focal is exclusive with it
    expect(section(store, "camera")).toEqual({});

    // a topic that cannot be rectified turns it off again
    await userEvent.click(topicRadio("/rect/image_raw"));
    expect(section(store, "rectify")).toBeUndefined();
    await userEvent.click(topicRadio("/cam/image_raw"));
    await userEvent.click(within(screen.getByRole("region", { name: "Rectification" })).getByRole("switch", { name: "Rectify the images" }));
    await userEvent.click(within(screen.getByRole("region", { name: "Rectification" })).getByRole("switch", { name: "Rectify the images" }));
    expect(section(store, "rectify")).toBeUndefined();
  });

  it("an MCAP whose topic has a single message renders without NaN", async () => {
    const single: VideoRecord = {
      ...mcapRecord,
      fps: 0,
      duration_s: 0,
      mcap: { durationS: 0, cameraInfoTopics: [], images: [{ ...mcapRecord.mcap!.images[1], rateHz: 0, count: 1, durationS: 0 }] },
    };
    await renderStudio(fakeVideoApi([single]));
    await openLibrary(/drive\.mcap/);
    const text = deck().textContent ?? "";
    expect(text).not.toMatch(/NaN|Infinity/);
    expect(within(deck()).getByTestId("segment-estimate")).toHaveTextContent("1 frame");
    expect(text).toContain("1 message");
  });

  it("the video client maps the topics JSON to camelCase", async () => {
    const raw = {
      duration_s: 1,
      camera_info_topics: ["/cam/camera_info"],
      images: [
        { name: "/cam/image_raw", schema: "sensor_msgs/msg/Image", width: 16, height: 12, encoding: "rgb8", rate_hz: 2, count: 3, duration_s: 1, camera_info_topic: "/cam/camera_info", has_camera_info: true, has_distortion: true, camera_info_error: null, supported: true, unsupported_reason: null },
      ],
    };
    const record = { ...mcapRecord, mcap: raw };
    const fetchImpl = async (url: RequestInfo | URL) =>
      new Response(JSON.stringify(String(url).endsWith("/topics") ? raw : String(url).endsWith("/api/videos") ? [record] : record), { status: 200 });
    const api = createVideoApi("", fetchImpl as typeof fetch);
    const expected = mcapRecord.mcap!.images[1];
    expect(await api.fetchTopics(mcapRecord.id)).toEqual({ durationS: 1, cameraInfoTopics: ["/cam/camera_info"], images: [expected] });
    expect((await api.getVideo(mcapRecord.id)).mcap?.images[0]).toEqual(expected);
    expect((await api.listVideos())[0].mcap?.images[0].rateHz).toBe(2);
  });
});
