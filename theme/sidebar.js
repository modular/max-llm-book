const body = document.body;
const sidebar_toggle = document.getElementById("sidebar-toggle-anchor");
const verticalToggle = document.getElementById("vertical-sidebar-toggle");
const isDesktop = window.innerWidth >= 1220;

let sidebar;
try {
  sidebar = localStorage.getItem("mdbook-sidebar");
} catch (e) {}

if (!sidebar || sidebar === "null") {
  sidebar = isDesktop ? "visible" : "hidden";
}

sidebar_toggle.checked = sidebar === "visible";
body.classList.remove("sidebar-visible", "sidebar-hidden");
body.classList.add("sidebar-" + sidebar);

function setAriaExpanded() {
  const isVisible = sidebar === "visible";
  document.getElementById("sidebar-toggle").setAttribute("aria-expanded", isVisible);
  verticalToggle.setAttribute("aria-expanded", isVisible);
}

function updateSidebarState(newState) {
  sidebar = newState;
  sidebar_toggle.checked = newState === "visible";
  body.classList.remove("sidebar-visible", "sidebar-hidden");
  body.classList.add("sidebar-" + sidebar);
  setAriaExpanded();
  updateTabIndices();
}

function updateTabIndices() {
  const tabIndex = sidebar === "visible" ? 0 : -1;
  Array.from(document.querySelectorAll("#sidebar a")).forEach(function (link) {
    link.setAttribute("tabIndex", tabIndex);
  });
}

window.addEventListener("resize", function () {
  const isNowDesktop = window.innerWidth >= 1220;
  const newState = isNowDesktop ? "visible" : "hidden";

  if (sidebar !== newState) {
    updateSidebarState(newState);
  }
});

const collapseSidebar = document.querySelector(".collapse-sidebar");
const newCollapse = collapseSidebar.cloneNode(true);
collapseSidebar.parentNode.replaceChild(newCollapse, collapseSidebar);

newCollapse.addEventListener(
  "click",
  function (event) {
    event.preventDefault();
    updateSidebarState("hidden");
    try {
      localStorage.setItem("mdbook-sidebar", "hidden");
    } catch (e) {}
  },
  true
);

const newVerticalToggle = verticalToggle.cloneNode(true);
verticalToggle.parentNode.replaceChild(newVerticalToggle, verticalToggle);

newVerticalToggle.addEventListener(
  "click",
  function (event) {
    event.preventDefault();
    const newState = sidebar === "visible" ? "hidden" : "visible";
    updateSidebarState(newState);
    try {
      localStorage.setItem("mdbook-sidebar", newState);
    } catch (e) {}
  },
  true
);

setAriaExpanded();
updateTabIndices();

// Prevent mdBook's book.js override on display state
const sidebarElement = document.getElementById("sidebar");
const styleObserver = new MutationObserver((mutations) => {
  mutations.forEach((mutation) => {
    if (mutation.attributeName === "style") {
      if (sidebarElement.style.display === "none") {
        sidebarElement.style.display = "";
      }
    }
  });
});

styleObserver.observe(sidebarElement, {
  attributes: true,
  attributeFilter: ["style"],
});
