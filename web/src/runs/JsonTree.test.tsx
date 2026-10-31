import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import { JsonTree } from "./JsonTree";

describe("JsonTree", () => {
  it("unknown metrics section renders in the generic tree", async () => {
    // A section no part of the UI knows about, as a future sink would add.
    const metrics = { frames_processed: 12, brand_new_section: { widget_score: 0.875, layers: ["a", "b"] }, flag: true, missing: null };
    render(<JsonTree value={metrics} label="metrics.json" />);
    const tree = screen.getByRole("list", { name: "metrics.json" });
    expect(tree).toHaveTextContent("frames_processed");
    expect(tree).toHaveTextContent("12");
    expect(screen.getByText("brand_new_section")).toBeInTheDocument();
    expect(screen.queryByText("widget_score")).toBeNull(); // collapsed: children not mounted
    await userEvent.click(screen.getByText("brand_new_section"));
    expect(screen.getByText("widget_score")).toBeInTheDocument();
    expect(screen.getByText("0.875")).toBeInTheDocument();
    expect(screen.getByText("[2 items]")).toBeInTheDocument();
    expect(tree).toHaveTextContent("null");
  });

  it("pages long lists", async () => {
    render(<JsonTree value={{ tracks: Array.from({ length: 250 }, (_, i) => i) }} />);
    await userEvent.click(screen.getByText("tracks"));
    expect(screen.getByText("Show 100 more of 150")).toBeInTheDocument();
    expect(document.querySelectorAll(".jtree__leaf")).toHaveLength(100);
    await userEvent.click(screen.getByText("Show 100 more of 150"));
    expect(document.querySelectorAll(".jtree__leaf")).toHaveLength(200);
  });
});
