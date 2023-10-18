$(".js-trigger-transition").on("click", function (e) {
  e.preventDefault();
  transition();
});

function transition() {
  var tl = new TimelineMax({
    onComplete: function () {
      // Redirect to index.html after animation completes
      window.location.href = "../index.html";
    },
  });

  tl.to(
    CSSRulePlugin.getRule("body:before"),
    0.2,
    { cssRule: { top: "50%" }, ease: Power2.easeOut },
    "close"
  )
    .to(
      CSSRulePlugin.getRule("body:after"),
      0.2,
      { cssRule: { bottom: "50%" }, ease: Power2.easeOut },
      "close"
    )
    .to($(".loader"), 0.2, { opacity: 1 })
    .to(
      CSSRulePlugin.getRule("body:before"),
      0.2,
      { cssRule: { top: "0%" }, ease: Power2.easeOut },
      "+=1.5",
      "open"
    )
    .to(
      CSSRulePlugin.getRule("body:after"),
      0.2,
      { cssRule: { bottom: "0%" }, ease: Power2.easeOut },
      "-=0.2",
      "open"
    )
    .to($(".loader"), 0.2, { opacity: 0 }, "-=0.2");
}

// Using a single function for handling the loader display
function showPage() {
  $("#loader").hide();
  $("#myDiv").css("display", "flex");
}

// Assuming you have a button with ID "button" for navigation
$("#button").on("click", function () {
  showPage();
  // Redirect to index.html after the animation completes
  window.location.href = "https://morochow.github.io/index.html";
});
