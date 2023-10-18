// Slider (all Slides in a container)
const slideri = document.querySelector(".slideri");
// All trails
const trail = document.querySelector(".trail").querySelectorAll(".div");

// Transform value
let value = 0;
// trail index number
let trailValue = 0;
// interval (Duration)
let interval = 4000;

// Function to slide forward
const slidei = (condition) => {
  // Clear interval
  clearInterval(start);
  // Update value and trailValue
  condition === "increase" ? initiateINC() : initiateDEC();
  // Move slide
  move(value, trailValue);
  // Restart Animation
  animate();
  // Start interval for slides back
  start = setInterval(() => slidei("increase"), interval);
};

// Function for increase (forward, next) configuration
const initiateINC = () => {
  // Remove active from all trails
  trail.forEach((cur) => cur.classList.remove("active"));
  // Increase transform value
  value = (value + 100 / trail.length) % 100; // Use modulo operator to loop back
  // Update trailValue based on value
  trailUpdate();
};

// Function for decrease (backward, previous) configuration
const initiateDEC = () => {
  // Remove active from all trails
  trail.forEach((cur) => cur.classList.remove("active"));
  // Decrease transform value
  value === 0
    ? (value = 100 - 100 / trail.length)
    : (value -= 100 / trail.length);
  // Update trailValue based on value
  trailUpdate();
};

// Function to transform slide
const move = (S, T) => {
  // Transform slider
  slideri.style.transform = `translateX(-${S}%)`;
  // Add active class to the current trail
  trail[T].classList.add("active");
};

// GSAP Animation
const tl = gsap.timeline({ defaults: { duration: 0.6, ease: "power2.inOut" } });
tl.from(".bg", { x: "-100%", opacity: 0 })
  .from("p", { opacity: 0 }, "-=0.3")
  .from("h1", { opacity: 0, y: "30px" }, "-=0.3")
  .from("button", { opacity: 0, y: "-40px" }, "-=0.8");

// Function to restart animation
const animate = () => tl.restart();

// Function to update trailValue based on slide value
const trailUpdate = () => {
  trailValue = Math.round(value / (100 / trail.length));
};

// Start interval for slides
let start = setInterval(() => slidei("increase"), interval);

// Next and Previous button function (SVG icon with different classes)
document.querySelectorAll("svg").forEach((cur) => {
  // Assign function based on the class Name("next" and "prev")
  cur.addEventListener("click", () =>
    cur.classList.contains("next") ? slidei("increase") : slidei("decrease")
  );
});

// Ensure that the first slide is active initially
trail[0].classList.add("active");

// Function to slide when trail is clicked
const clickCheck = (e) => {
  // Clear interval
  clearInterval(start);
  // Remove active class from all trails
  trail.forEach((cur) => cur.classList.remove("active"));
  // Get selected trail
  const check = e.target;
  // Add active class
  check.classList.add("active");

  // Update slide value based on the selected trail
  const trailIndex = Array.from(trail).indexOf(check);
  value = (100 / trail.length) * trailIndex;
  // Update trail based on value
  trailUpdate();
  // Transform slide
  move(value, trailValue);
  // Start animation
  animate();
  // Start interval
  start = setInterval(() => slidei("increase"), interval);
};

// Add function to all trails
trail.forEach((cur) => cur.addEventListener("click", (ev) => clickCheck(ev)));

// Mobile touch Slide Section
const touchSlide = (() => {
  let start, move, change, sliderWidth;

  // Do this on initial touch on the screen
  slideri.addEventListener("touchstart", (e) => {
    // Get the touch position of X on the screen
    start = e.touches[0].clientX;
    // The width of the slider container divided by the number of slides
    sliderWidth = (slideri.clientWidth / trail.length) * 100; // Convert to percentage
  });

  // Do this on touchDrag on the screen
  slideri.addEventListener("touchmove", (e) => {
    // Prevent default function
    e.preventDefault();
    // Get the touch position of X on the screen when dragging stops
    move = e.touches[0].clientX;
    // Subtract initial position from end position and save to change variable
    change = start - move;
  });

  const mobile = (e) => {
    // If change is greater than a quarter of sliderWidth, next else do nothing
    change > sliderWidth / 4 ? slidei("increase") : null;
    // If change * -1 is greater than a quarter of sliderWidth, prev else do nothing
    change * -1 > sliderWidth / 4 ? slidei("decrease") : null;
    // Reset all variables to 0
    [start, move, change, sliderWidth] = [0, 0, 0, 0];
  };

  // Call mobile on touch end
  slideri.addEventListener("touchend", mobile);
})();