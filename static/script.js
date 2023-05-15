const uploadInput = document.querySelector('input[name="image-upload"]');
const uploadedImage = document.querySelector(".original-image");
const filteredImage = document.querySelector(".filtered-image");
const arrow = document.querySelector(".arrow");
const applyFilterButton = document.querySelector(".dropbtn");
const filterSelect = document.querySelector(".filter-select");
const kernelSizeSlider = document.querySelector("#kernel-size");
const kernelSizeValue = document.querySelector("#kernel-size-value");
kernelSizeValue.style.textAlign = "left";

function createSlider(id, min, max, value) {
  let slider = document.createElement("input");
  slider.setAttribute("type", "range");
  slider.setAttribute("min", min);
  slider.setAttribute("max", max);
  slider.setAttribute("value", value);
  slider.setAttribute("id", id);
  return slider;
}

function createLabel(forAttr, innerHtml) {
  let label = document.createElement("label");
  label.setAttribute("for", forAttr);
  label.innerHTML = innerHtml;
  label.style.textAlign = "center";
  return label;
}

function createValueSpan(id, value) {
  let valueSpan = document.createElement("span");
  valueSpan.setAttribute("id", id);
  valueSpan.innerHTML = value;
  valueSpan.style.marginBottom = "0.2em";
  return valueSpan;
}

function appendChildren(parent, children) {
  children.forEach((child) => parent.appendChild(child));
}

function createRadioButton(radioContainer, name, value, innerHTML) {
  // Create a radio button and its label
  var radioBtn = document.createElement("input");
  radioBtn.type = "radio";
  radioBtn.name = name;
  radioBtn.value = value;

  var label = document.createElement("label");
  label.innerHTML = innerHTML;
  label.style.marginLeft = "0.4em";
  label.addEventListener("click", function () {
    radioBtn.checked = true; // Set the radio button as checked
  });

  // Create a container div for the radio button and label
  var container = document.createElement("div");
  container.style.display = "flex"; // Display as flex
  container.style.alignItems = "center"; // Align items vertically in the center
  container.style.marginBottom = "10px"; // Add space between the radio button and label
  container.appendChild(radioBtn);
  container.appendChild(label);

  // Append the container to the radio container div
  radioContainer.appendChild(container);
}

var kernelOptions = {
  SobelOperator: [
    { value: "0", label: "[[-1,-2,-1],[0,0,0],[1,2,1]]" },
    { value: "1", label: "[[-1,0,-1],[-2,0,2],[-1,0,1]]" },
  ],
  LaplaceOperator: [
    { value: "0", label: "[[0,1,0],[1,-4,1],[0,1,0]]" },
    { value: "1", label: "[[0,-1,0],[-1,4,-1],[0,-1,0]]" },
    { value: "2", label: "[[1,1,1],[1,-8,1],[1,1,1]]" },
    { value: "3", label: "[[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]" },
  ],
};


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
      if (filterSelect.value === "UnsharpAvgFilter") {
        min = 0;
        max = 1;
        value = 0;
      } else {
        min = 1;
        max = 100;
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
    }else if (filterSelect.value == "apply_uniform_noise") {
      let extraParametersDiv = document.querySelector(".extra-parameters");


        let min = 0;
        let max = 255;
        let valueMin = 0;
        let valueMax = 255;

        // Create a containers for the sliders and labels
        let containerMin = document.createElement("div");
        let containerMax = document.createElement("div");

        containerMin.style.display = "flex";
        containerMin.style.flexDirection = "column";
        containerMin.style.width = "100%";

        containerMax.style.display = "flex";
        containerMax.style.flexDirection = "column";
        containerMax.style.width = "100%";

        // Create the slider elements
        let sliderMin = createSlider("min-slider", min, max, valueMin);
        let sliderMax = createSlider("max-slider", min, max, valueMax);

        // Create the labels for the sliders
        let labelMin = createLabel("min-slider", "Min:");
        let labelMax = createLabel("max-slider", "Max:");

        // Create the span elements to display the slider values
        let valueSpanMin = createValueSpan("min-value", valueMin);
        let valueSpanMax = createValueSpan("max-value", valueMax);

        // Append the sliders and labels to the containers
        appendChildren(containerMin, [labelMin, sliderMin, valueSpanMin]);
        appendChildren(containerMax, [labelMax, sliderMax, valueSpanMax]);

        // Insert the containers into the extra-parameters div
        appendChildren(extraParametersDiv, [containerMin, containerMax]);

        // Make the extra-parameters div visible
        extraParametersDiv.style.visibility = "visible";

        // Add event listeners to the sliders to update value display when the sliders are changed
        sliderMin.addEventListener("input", function () {
          document.querySelector("#min-value").innerHTML = this.value;
        });

        sliderMax.addEventListener("input", function () {
          document.querySelector("#max-value").innerHTML = this.value;
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
    } else if (filterSelect.value in kernelOptions) {
      var kernelLabel = document.createElement("label");
      kernelLabel.innerHTML = "Choose Kernel";
      extraParametersDiv.appendChild(kernelLabel);

      // Create a div to contain the radio buttons
      var radioContainer = document.createElement("div");
      extraParametersDiv.appendChild(radioContainer);

      kernelOptions[filterSelect.value].forEach((option) => {
        createRadioButton(
          radioContainer,
          "kernelOption",
          option.value,
          option.label
        );
      });
    } else if (filterSelect.value === "apply_gaussian_noise") {
      let extraParametersDiv = document.querySelector(".extra-parameters"); // Assuming this is your div

      ["mean", "std"].forEach((param) => {
        let container = document.createElement("div");
        container.style.display = "flex";
        container.style.flexDirection = "column";
        container.style.width = "100%";

        // Create the slider element
        let slider = document.createElement("input");
        slider.setAttribute("type", "range");
        slider.setAttribute("min", param === "mean" ? "-50" : "1");
        slider.setAttribute("max", param === "mean" ? "50" : "100");
        slider.setAttribute("step", "0.01");
        slider.setAttribute("value", param === "mean" ? "0" : "20");
        slider.setAttribute("id", `${param}-slider`);

        // Create the label for the slider
        let label = document.createElement("label");
        label.setAttribute("for", `${param}-slider`);
        label.innerHTML = `${param}:`;
        label.style.textAlign = "center";

        // Create the span element to display the slider value
        let valueSpan = document.createElement("span");
        valueSpan.setAttribute("id", `${param}-value`);
        valueSpan.innerHTML = slider.getAttribute("value");
        valueSpan.style.marginBottom = "0.2em";

        // Append the slider and labels to the container
        container.appendChild(label);
        container.appendChild(slider);
        container.appendChild(valueSpan);

        // Insert the container into the extra-parameters div
        extraParametersDiv.appendChild(container);

        // Add event listener to the slider to update value display when the slider is changed
        slider.addEventListener("input", function () {
          document.querySelector(`#${param}-value`).innerHTML = this.value;
        });
      });
    }
  });

applyFilterButton.addEventListener("click", async function () {
  const filterType = filterSelect.value;
  const kernelSize = kernelSizeSlider.value;
  kernelSizeValue.textContent = kernelSize;
  console.log(filterType);

  let reuseFilteredImageCheckbox = document.querySelector(
    "#reuse-filtered-image"
  );
  let reuseFilteredImage = reuseFilteredImageCheckbox.checked;

  const extraValuesArray = [];

  const extraValuesSpans = document.querySelectorAll(".extra-parameters span");
  const extraValuesRadios = document.querySelectorAll(
    'input[name="kernelOption"]'
  );

  console.log(extraValuesRadios);
  extraValuesSpans.forEach((span) => {
    extraValuesArray.push(span.innerHTML);
  });

  extraValuesRadios.forEach((radio) => {
    radio.checked && extraValuesArray.push(radio.value);
  });

  console.log(extraValuesArray);

  let inputImageData = reuseFilteredImage
    ? filteredImage.src
    : uploadedImage.src;

  const body = `image_data=${encodeURIComponent(
    inputImageData
  )}&filter_type=${filterType}&kernel_size=${kernelSize}&extraParams=${extraValuesArray.join(
    ","
  )}`;

  const response = await fetch("/process-image", {
    method: "POST",
    headers: {
      "Content-Type": "application/x-www-form-urlencoded",
    },
    body: body,
  });

  const result = await response.json();
  filteredImage.src = result.processed_image;
  uploadedImage.src = result.original_greyed;
});

kernelSizeSlider.addEventListener("input", function () {
  const kernelSize = kernelSizeSlider.value;
  kernelSizeValue.textContent = kernelSize;
});
