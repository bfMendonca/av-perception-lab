import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { Field } from "../api/types";
import { thresholdField } from "../test/fixtures";
import { FieldList, fieldControls } from "./fieldControls";
import { registerFieldControl } from "./registry";

const unregister: (() => void)[] = [];
afterEach(() => unregister.splice(0).forEach((u) => u()));

describe("fieldControls", () => {
  it("fieldControls renders a slider for a float with hint range", () => {
    const onChange = vi.fn();
    render(fieldControls(thresholdField, { value: 0.5, onChange, path: ["d", 0, "threshold"] }));
    const slider = screen.getByRole("slider");
    expect(slider).toHaveAttribute("min", "0");
    expect(slider).toHaveAttribute("max", "1");
    expect(slider).toHaveAttribute("step", "0.05");
    fireEvent.change(slider, { target: { value: "0.7" } });
    expect(onChange).toHaveBeenLastCalledWith(0.7);
    expect(screen.getByText("Score threshold")).toBeInTheDocument();
  });

  it("fieldControls renders a select for enum with choices", async () => {
    const field: Field = { name: "variant", type: "enum", choices: ["small", "large"], default: "large", required: false, nullable: false };
    const onChange = vi.fn();
    render(fieldControls(field, { value: "large", onChange, path: ["variant"] }));
    const select = screen.getByRole("combobox");
    expect([...(select as HTMLSelectElement).options].map((o) => o.value)).toEqual(["small", "large"]);
    await userEvent.selectOptions(select, "small");
    expect(onChange).toHaveBeenCalledWith("small");
  });

  it("advanced fields are collapsed by default", async () => {
    const fields: Field[] = [
      { name: "accel_std", type: "float", default: 2, required: false, nullable: false, hint: { label: "Acceleration noise" } },
      { name: "tau_s", type: "float", default: 0.1, required: false, nullable: false, hint: { label: "EMA time constant", advanced: true } },
    ];
    render(<FieldList fields={fields} value={{}} basePath={["smoothing"]} onChange={() => {}} />);
    expect(screen.getByText("Acceleration noise")).toBeInTheDocument();
    expect(screen.queryByText("EMA time constant")).not.toBeInTheDocument();
    const toggle = screen.getByRole("button", { name: /advanced/i });
    expect(toggle).toHaveAttribute("aria-expanded", "false");
    await userEvent.click(toggle);
    expect(screen.getByText("EMA time constant")).toBeInTheDocument();
  });

  it("registerFieldControl overrides the default control for a matching field", () => {
    unregister.push(
      registerFieldControl(
        (f) => f.hint?.widget === "knob",
        ({ value }) => <output data-testid="knob">{String(value)}</output>,
      ),
    );
    const knob: Field = { ...thresholdField, hint: { ...thresholdField.hint, widget: "knob" } };
    render(fieldControls(knob, { value: 0.3, onChange: () => {}, path: ["x"] }));
    expect(screen.getByTestId("knob")).toHaveTextContent("0.3");
    expect(screen.queryByRole("slider")).not.toBeInTheDocument();
  });

  it("unknown field type falls back to a raw YAML text editor with validation", async () => {
    const field: Field = { name: "warp", type: "tensor-field", default: null, required: false, nullable: true };
    const onChange = vi.fn();
    render(fieldControls(field, { value: { gain: 2 }, onChange, path: ["warp"] }));
    const editor = screen.getByRole("textbox", { name: /warp/i });
    expect(editor).toHaveValue("gain: 2");
    await userEvent.clear(editor);
    await userEvent.type(editor, "gain: [[1"); // "[[" types a literal "["
    await userEvent.tab();
    expect(screen.getByRole("alert")).toHaveTextContent(/invalid yaml/i);
    expect(onChange).not.toHaveBeenCalled();
    await userEvent.clear(editor);
    await userEvent.type(editor, "gain: 3");
    await userEvent.tab();
    expect(onChange).toHaveBeenCalledWith({ gain: 3 });
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });
});
