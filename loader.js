// Function to handle the transition
const transition = () => {
  const tl = new TimelineMax();

  tl.to(CSSRulePlugin.getRule("body:before"), 0.2, { cssRule: { top: "50%" }, ease: Power2.easeOut }, "close")
    .to(CSSRulePlugin.getRule("body:after"), 0.2, { cssRule: { bottom: "50%" }, ease: Power2.easeOut }, "close")
    .to($(".loader"), 0.2, { opacity: 1 })
    .to(CSSRulePlugin.getRule("body:before"), 0.2, { cssRule: { top: "0%" }, ease: Power2.easeOut }, "+=1.5", "open")
    .to(CSSRulePlugin.getRule("body:after"), 0.2, { cssRule: { bottom: "0%" }, ease: Power2.easeOut }, "-=0.2", "open")
    .to($(".loader"), 0.2, { opacity: 0 }, "-=0.2");
};

// Function to show the page after loading
const showPage = () => {
  $("#loader").hide();
  $("#myDiv").css("display", "flex");
};

// Click event for the button with class "js-trigger-transition"
$(".js-trigger-transition").on("click", (e) => {
  e.preventDefault();
  transition();
});

// Click event for the button with ID "button"
$("#button").on("click", () => {
  showPage();
  // Add more logic here if needed
  window.location.href = "https://morchow.github.io/index.html";
});

