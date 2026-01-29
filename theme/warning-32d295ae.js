(function () {
    "use strict";
    function moveWarnings() {
      const warnings = document.querySelectorAll(".warning");
      if (warnings.length === 0) {
        return;
      }
      const pageDiv = document.querySelector(".page");
      if (!pageDiv) {
        return;
      }
      const parent = pageDiv.parentNode;
      if (!parent) {
        return;
      }
      warnings.forEach(function (warning) {
        parent.insertBefore(warning, pageDiv);
      });
    }
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", moveWarnings);
    } else {
      moveWarnings();
    }
  })();