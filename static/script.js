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
    if (filterSelect.value === "UnsharpAvgFilter") {
      // When an option is selected, create a new slider and labels
      let container = document.createElement("div");
      container.style.display = "flex";
      container.style.flexDirection = "column";
      container.style.width = "100%";

      let slider = document.createElement("input");
      slider.setAttribute("type", "range");
      slider.setAttribute("min", "0");
      slider.setAttribute("max", "1");
      slider.setAttribute("step", "0.01");
      slider.setAttribute("value", "0");
      slider.setAttribute("id", "k-slider");

      let label = document.createElement("label");
      label.setAttribute("for", "k-slider");
      label.innerHTML = "k:";
      label.style.textAlign = "center";

      let valueSpan = document.createElement("span");
      valueSpan.setAttribute("id", "k-value");
      valueSpan.innerHTML = "0";
      valueSpan.style.marginBottom = "0.2em";
      // Append the slider and labels to the container
      container.appendChild(label);
      container.appendChild(slider);
      container.appendChild(valueSpan);

      // Insert the container into the extra-parameters div
      let extraParametersDiv = document.querySelector(".extra-parameters");
      extraParametersDiv.innerHTML = "";
      extraParametersDiv.appendChild(container);

      // Make the extra-parameters div visible
      extraParametersDiv.style.visibility = "visible";

      // Add event listener to slider to update value display when the slider is changed
      slider.addEventListener("input", function () {
        document.querySelector("#k-value").innerHTML = this.value;
      });
    }
  });



applyFilterButton.addEventListener("click", async function () {
  const filterType = filterSelect.value;
  const kernelSize = kernelSizeSlider.value;
  kernelSizeValue.textContent = kernelSize;
  console.log(filterType);

  const extraValuesArray = [];

  const extraValues = document.querySelectorAll(".extra-parameters span");
  extraValues.forEach((span) => {
    extraValuesArray.push(span.innerHTML);
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
