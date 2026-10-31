import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import type { MetricInfo } from "../api/runs";
import { ConfigDiff, KpiCards } from "./CompareView";
import { deltaView } from "./format";

// Metric descriptors as the server sends them (test data: the component
// itself knows no metric names).
const metrics: MetricInfo[] = [
  { key: "jitter_m", label: "Jitter", unit: "m", better: "lower" },
  { key: "fps", label: "Throughput", unit: "frames/s", better: "higher" },
  { key: "median_distance_m", label: "Median distance", unit: "m", better: null },
  { key: "absent", label: "Absent", unit: null, better: null },
];

describe("KpiCards", () => {
  it("deltas show the sign and a good/bad class for jitter", () => {
    const { container } = render(
      <KpiCards
        metrics={metrics}
        summary={{ jitter_m: 0.42, fps: 2.1, median_distance_m: 23.4, absent: null }}
        deltas={{ jitter_m: { abs: -0.09, rel: -0.176 }, fps: { abs: -0.2, rel: -0.087 }, median_distance_m: { abs: 0.5, rel: 0.022 } }}
        comparing
      />,
    );
    const jitter = screen.getByText("Jitter").closest(".kpi") as HTMLElement;
    expect(jitter).toHaveTextContent("0.42 m");
    const delta = jitter.querySelector(".kpi__delta") as HTMLElement;
    expect(delta).toHaveTextContent("\u221218 %");
    expect(delta).toHaveClass("kpi__delta--good"); // lower jitter is better
    const fps = screen.getByText("Throughput").closest(".kpi") as HTMLElement;
    expect(fps.querySelector(".kpi__delta")).toHaveClass("kpi__delta--bad");
    const median = screen.getByText("Median distance").closest(".kpi") as HTMLElement;
    expect(median.querySelector(".kpi__delta")).toHaveTextContent("+2.2 %");
    expect(median.querySelector(".kpi__delta")).toHaveClass("kpi__delta--neutral");
    expect(screen.queryByText("Absent")).toBeNull(); // neither run has it
    expect(container.querySelectorAll(".kpi")).toHaveLength(3);
  });

  it("formats a worse jitter and an absolute delta when the base is zero", () => {
    expect(deltaView(metrics[0], { abs: 0.1, rel: 0.25 })).toMatchObject({ text: "+25 %", tone: "bad" });
    expect(deltaView(metrics[0], { abs: 0.3, rel: null })).toMatchObject({ text: "+0.3 m", tone: "bad" });
    expect(deltaView(metrics[0], { abs: 0, rel: 0 })?.tone).toBe("neutral");
  });
});

describe("ConfigDiff", () => {
  it("hides the keys the server sets until asked", async () => {
    render(
      <ConfigDiff
        diff={[
          { path: "smoothing.accel_std", a: 2, b: 3.5, change: "changed", server_set: false },
          { path: "output.tag", a: "web-a", b: "web-b", change: "changed", server_set: true },
        ]}
      />,
    );
    const body = screen.getByLabelText("Configuration changes");
    expect(body).toHaveTextContent("- smoothing.accel_std: 2");
    expect(body).toHaveTextContent("+ smoothing.accel_std: 3.5");
    expect(body).not.toHaveTextContent("output.tag");
    await userEvent.click(screen.getByRole("button", { name: /Show 1 key set by the server/ }));
    expect(screen.getByLabelText("Configuration changes")).toHaveTextContent("+ output.tag: web-b");
  });

  it("explains a run without a recorded configuration", () => {
    render(<ConfigDiff diff={null} origin={{ a: null, b: "config.yaml" }} />);
    expect(screen.getByText(/The comparison run recorded no configuration/)).toBeInTheDocument();
  });
});
