const uploadInput = document.querySelector('input[name="image-upload"]');
const uploadedImage = document.querySelector(".original-image");
const filteredImage = document.querySelector(".filtered-image");
const arrow = document.querySelector(".arrow");
const applyFilterButton = document.querySelector(".dropbtn");
const filterSelect = document.querySelector(".filter-select");
const kernelSizeSlider = document.querySelector("#kernel-size");
const kernelSizeValue = document.querySelector("#kernel-size-value");
kernelSizeValue.style.textAlign = 'left';
uploadInput.addEventListener("change", function () {
  const file = this.files[0];
  const reader = new FileReader();
  reader.addEventListener("load", function () {
    uploadedImage.src = reader.result;
    uploadedImage.removeAttribute("hidden");

    filteredImage.src = reader.result;
    filteredImage.removeAttribute("hidden");

    arrow.removeAttribute("hidden");
  });
  reader.readAsDataURL(file);
});


document
  .querySelector(".filter-select")
  .addEventListener("change", function () {
    let extraParametersDiv = document.querySelector(".extra-parameters");
// Clear existing content in extraParametersDiv
extraParametersDiv.innerHTML = "";

if (
  filterSelect.value === "UnsharpAvgFilter" ||
  filterSelect.value === "HighboostFilter"
) {

  let min = 0;
  let max = 1;
  let value = 0;
  if(filterSelect.value === "UnsharpAvgFilter"){
      min = 0;
      max = 1;
      value = 0;
  }
  else{
      min = 1;
      max = 5;
      value = 1;
  }
  // Create a container for the slider and labels
  let container = document.createElement("div");
  container.style.display = "flex";
  container.style.flexDirection = "column";
  container.style.width = "100%";

  // Create the slider element
  let slider = document.createElement("input");
  slider.setAttribute("type", "range");
  slider.setAttribute("min", min);
  slider.setAttribute("max", max);
  slider.setAttribute("step", "0.01");
  slider.setAttribute("value", value);
  slider.setAttribute("id", "k-slider");

  // Create the label for the slider
  let label = document.createElement("label");
  label.setAttribute("for", "k-slider");
  label.innerHTML = "k:";
  label.style.textAlign = "center";

  // Create the span element to display the slider value
  let valueSpan = document.createElement("span");
  valueSpan.setAttribute("id", "k-value");
  valueSpan.innerHTML = value;
  valueSpan.style.marginBottom = "0.2em";

  // Append the slider and labels to the container
  container.appendChild(label);
  container.appendChild(slider);
  container.appendChild(valueSpan);

  // Insert the container into the extra-parameters div
  extraParametersDiv.appendChild(container);

  // Make the extra-parameters div visible
  extraParametersDiv.style.visibility = "visible";

  // Add event listener to the slider to update value display when the slider is changed
  slider.addEventListener("input", function () {
    document.querySelector("#k-value").innerHTML = this.value;
  });
} else if (filterSelect.value === "RobertCrossGradient") {
  var kernelLabel = document.createElement("label");
  kernelLabel.innerHTML = "Choose Kernel";
  extraParametersDiv.appendChild(kernelLabel);

  // Create a div to contain the radio buttons
  var radioContainer = document.createElement("div");
  extraParametersDiv.appendChild(radioContainer);

  // Create the first radio button [1,0] and its label
  var radioBtn1 = document.createElement("input");
  radioBtn1.type = "radio";
  radioBtn1.name = "kernelOption";
  radioBtn1.value = "0";

  var label1 = document.createElement("label");
  label1.innerHTML = "[[0,-1],[1,0]]";
  label1.style.marginLeft = "0.4em";

  label1.addEventListener("click", function () {
    radioBtn1.checked = true; // Set the radio button as checked
  });

  // Create a container div for the radio button and label
  var container1 = document.createElement("div");
  container1.style.display = "flex"; // Display as flex
  container1.style.alignItems = "center"; // Align items vertically in the center
  container1.style.marginBottom = "10px"; // Add space between the radio button and label
  container1.appendChild(radioBtn1);
  container1.appendChild(label1);

  // Append the container to the radio container div
  radioContainer.appendChild(container1);

  // Create the second radio button [0,1] and its label
  var radioBtn2 = document.createElement("input");
  radioBtn2.type = "radio";
  radioBtn2.name = "kernelOption";
  radioBtn2.value = "1";

  var label2 = document.createElement("label");
  label2.innerHTML = "[[-1,0],[0,1]]";
  label2.style.marginLeft = "0.4em";
  label2.addEventListener("click", function () {
    radioBtn2.checked = true; // Set the radio button as checked
  });

  // Create a container div for the radio button and label
  var container2 = document.createElement("div");
  container2.style.display = "flex"; // Display as flex
  container2.style.alignItems = "center"; // Align items vertically in the center
  container2.style.marginBottom = "10px"; // Add space between the radio button and label
  container2.appendChild(radioBtn2);
  container2.appendChild(label2);

  // Append the container to the radio container div
  radioContainer.appendChild(container2);
}
});



applyFilterButton.addEventListener("click", async function () {
  const filterType = filterSelect.value;
  const kernelSize = kernelSizeSlider.value;
  kernelSizeValue.textContent = kernelSize;
  console.log(filterType);

  const extraValuesArray = [];

  const extraValuesSpans = document.querySelectorAll(".extra-parameters span");
  const extraValuesRadios = document.querySelectorAll('input[name="kernelOption"]');

  console.log(extraValuesRadios);
  extraValuesSpans.forEach((span) => {
    extraValuesArray.push(span.innerHTML);
  });
  
  extraValuesRadios.forEach((radio) => {
    radio.checked && extraValuesArray.push(radio.value);
  });



  console.log(extraValuesArray);


  const body = `image_data=${encodeURIComponent(
    uploadedImage.src
  )}&filter_type=${filterType}&kernel_size=${kernelSize}&extraParams=${extraValuesArray.join(',')}`;

  const response = await fetch("/process-image", {
    method: "POST",
    headers: {
      "Content-Type": "application/x-www-form-urlencoded",
    },
    body: body,
  });

  const result = await response.json();
  filteredImage.src = result.processed_image;
});

kernelSizeSlider.addEventListener("input", function () {
  const kernelSize = kernelSizeSlider.value;
  kernelSizeValue.textContent = kernelSize;
});
