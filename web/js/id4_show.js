import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

// Shows the generated Ideogram-4 JSON inside the node (read-only textarea),
// so no external "Show Text" pack is needed to inspect the output.
app.registerExtension({
  name: "Ideogram4.ShowJson",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "Id4JsonPromptFromImage") return;

    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      onExecuted?.apply(this, arguments);

      // drop a previous preview widget so it doesn't stack on re-run
      if (this.widgets) {
        const i = this.widgets.findIndex((w) => w.name === "id4_preview");
        if (i !== -1) {
          this.widgets[i].onRemove?.();
          this.widgets.splice(i, 1);
        }
      }

      const w = ComfyWidgets["STRING"](
        this,
        "id4_preview",
        ["STRING", { multiline: true }],
        app
      ).widget;
      w.inputEl.readOnly = true;
      w.inputEl.style.opacity = 0.85;
      w.value = (message?.text || []).join("");

      requestAnimationFrame(() => {
        const sz = this.computeSize();
        this.onResize?.(sz);
        app.graph.setDirtyCanvas(true, false);
      });
    };
  },
});
