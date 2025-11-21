function createCopyButtons() {
  const codeBlocks = document.querySelectorAll("pre > code");

  codeBlocks.forEach((code, index) => {
    const pre = code.parentElement;

    if (pre.querySelector(".buttons")) {
      return;
    }

    const buttons = document.createElement("div");
    buttons.className = "buttons";

    const clipButton = document.createElement("button");
    clipButton.className = "clip-button copy-btn-${index}";
    clipButton.title = "Copy to clipboard";
    clipButton.setAttribute("aria-label", "Copy to clipboard");

    buttons.appendChild(clipButton);
    pre.insertBefore(buttons, pre.firstChild);
  });

  return codeBlocks.length;
}

function setupClipboard() {
  if (typeof ClipboardJS === "undefined") {
    setTimeout(setupClipboard, 100);
    return;
  }

  const clipboard = new ClipboardJS(".clip-button", {
    text: function (trigger) {
      const pre = trigger.closest("pre");
      const code = pre.querySelector("code");
      return code.innerText;
    },
  });

  clipboard.on("success", function (e) {
    e.clearSelection();
    const button = e.trigger;
    button.classList.add("tooltipped");

    setTimeout(() => {
      button.classList.remove("tooltipped");
    }, 1000);
  });

  clipboard.on("error", function (e) {
    const button = e.trigger;
    button.classList.add("tooltipped");
    console.error("Copy failed:", e);

    setTimeout(() => {
      button.classList.remove("tooltipped");
    }, 1000);
  });
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", () => {
    createCopyButtons();
    setupClipboard();
  });
} else {
  createCopyButtons();
  setupClipboard();
}
